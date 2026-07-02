from typing import Callable, Optional, Any, Union
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are


def quadratic_cost_and_penalty(
    Q: jax.Array, 
    R: jax.Array,
    x_ref: Optional[jax.Array] = None, 
    u_ref: Optional[jax.Array] = None,
    x_min: Optional[jax.Array] = None, 
    x_max: Optional[jax.Array] = None,
    u_min: Optional[jax.Array] = None, 
    u_max: Optional[jax.Array] = None,
    w_viol_lin: float = 0.0, 
    w_viol_quad: float = 0.0
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array,jax.Array,jax.Array]]:
    """
    Creates a trajectory cost function with quadratic tracking and soft constraints.
    
    All reference and bound arguments can be either static (e.g., shape (NX,)) 
    or time-varying (e.g., shape (T, NX)).
    
    Parameters
    ----------
    Q : jax.Array
        State cost matrix of shape (NX, NX) or (T, NX, NX).
    R : jax.Array
        Input cost matrix of shape (NU, NU) or (T, NU, NU).
    x_ref : jax.Array, optional
        Target state trajectory. Assumes origin if None.
    u_ref : jax.Array, optional
        Target input trajectory. Assumes origin if None.
    x_min, x_max : jax.Array, optional
        Lower and upper state bounds. Inf/NaN are handled safely.
    u_min, u_max : jax.Array, optional
        Lower and upper input bounds. Inf/NaN are handled safely.
    w_viol_lin : float, default=0.0
        Weight for the linear (L1) penalty on constraint violations.
    w_viol_quad : float, default=0.0
        Weight for the quadratic (L2) penalty on constraint violations.
        
    Returns
    -------
    Callable[[jax.Array, jax.Array], tuple[jax.Array,jax.Array,jax.Array]]
        A function `cost_fn(xs, us)` returning the scalar cost, the tracking
        cost, and the constraint violation (l1).
    """
    
    def cost_fn(xs: jax.Array, us: jax.Array) -> tuple[jax.Array,jax.Array,jax.Array]:
        # 1. Compute tracking errors (broadcasting handles time-varying vs static)
        dx = xs - x_ref if x_ref is not None else xs
        du = us - u_ref if u_ref is not None else us

        # 2. Compute Quadratic Tracking Cost
        if Q.ndim == 2:
            track_x = jnp.einsum("ti,ij,tj->", dx, Q, dx)
        else:
            track_x = jnp.einsum("ti,tij,tj->", dx, Q, dx)
            
        if R.ndim == 2:
            track_u = jnp.einsum("ti,ij,tj->", du, R, du)
        else:
            track_u = jnp.einsum("ti,tij,tj->", du, R, du)

        track_cost = track_x + track_u

        # 3. Compute Violations (Clipping to 0 for constraints satisfied)
        viol_x = 0.0
        if x_max is not None:
            viol_x += jnp.where(jnp.isfinite(x_max), jnp.maximum(0.0, xs - x_max), 0.0)
        if x_min is not None:
            viol_x += jnp.where(jnp.isfinite(x_min), jnp.maximum(0.0, x_min - xs), 0.0)
            
        viol_u = 0.0
        if u_max is not None:
            viol_u += jnp.where(jnp.isfinite(u_max), jnp.maximum(0.0, us - u_max), 0.0)
        if u_min is not None:
            viol_u += jnp.where(jnp.isfinite(u_min), jnp.maximum(0.0, u_min - us), 0.0)

        # 4. Apply linear and quadratic violation penalties
        penalty_cost = 0.0
        if w_viol_lin > 0.0:
            penalty_cost += w_viol_lin * (jnp.sum(viol_x) + jnp.sum(viol_u))
        if w_viol_quad > 0.0:
            penalty_cost += w_viol_quad * (jnp.sum(viol_x**2) + jnp.sum(viol_u**2))

        return track_cost + penalty_cost, track_cost, jnp.sum(viol_x) + jnp.sum(viol_u)

    return cost_fn


def dare_init_theta(
    plant: Any, 
    Q: Union[jax.Array, np.ndarray], 
    R: Union[jax.Array, np.ndarray], 
    x0: Optional[jax.Array] = None, 
    u0: Optional[jax.Array] = None, 
    nx: Optional[int] = None,
    nu: Optional[int] = None,
    plant_params: Optional[Any] = None,
    reg: float = 0.0
) -> jax.Array:
    """
    Initializes the lower-triangular Cholesky elements (c_q) of the 
    Discrete Algebraic Riccati Equation (DARE) solution for a given plant.

    Parameters
    ----------
    plant : Any
        The dynamics object containing a `step` method.
    Q : jax.Array or np.ndarray
        State cost matrix of shape (NX, NX).
    R : jax.Array or np.ndarray
        Control effort cost matrix of shape (NU, NU).
    x0 : jax.Array, optional
        The state operating point for linearization. Defaults to zeros if None.
    u0 : jax.Array, optional
        The control input operating point for linearization. Defaults to zeros if None.
    nx : int, optional
        State dimension. Required if x0 is not provided and plant lacks an 'nx' attribute.
    nu : int, optional
        Input dimension. Required if u0 is not provided and plant lacks a 'nu' attribute.
    plant_params : Any, optional
        Parameters passed dynamically to the plant dynamics step function.
    reg : float, default=0.0
        Terminal cost regularization factor subtracted from the identity matrix.

    Returns
    -------
    jax.Array
        A flattened array of the lower triangular elements of the 
        regularized DARE solution's Cholesky decomposition.
    """
    # 1. Resolve State Dimension (nx) and Initial State Operating Point (x0)
    if x0 is not None:
        n_x = x0.shape[0]
    else:
        n_x = nx if nx is not None else getattr(plant, "nx", None)
        if n_x is None:
            raise ValueError("State dimension could not be inferred. Provide 'x0', 'nx', or ensure 'plant.nx' exists.")
        x0 = jnp.zeros((n_x,))

    # 2. Resolve Input Dimension (nu) and Initial Control Operating Point (u0)
    if u0 is not None:
        n_u = u0.shape[0]
    else:
        n_u = nu if nu is not None else getattr(plant, "nu", None)
        if n_u is None:
            raise ValueError("Input dimension could not be inferred. Provide 'u0', 'nu', or ensure 'plant.nu' exists.")
        u0 = jnp.zeros((n_u,))

    # 3. Define an isolated local step wrapper to cleanly route optional plant parameters
    if plant_params is not None:
        def step_local(x: jax.Array, u: jax.Array) -> jax.Array:
            return plant.step(x, u, plant_params)
    else:
        def step_local(x: jax.Array, u: jax.Array) -> jax.Array:
            return plant.step(x, u)

    # 4. Linearize the dynamics around (x0, u0) to get state-space matrices
    A = np.array(jax.jacfwd(step_local, argnums=0)(x0, u0))
    B = np.array(jax.jacfwd(step_local, argnums=1)(x0, u0))

    # 5. Solve the DARE to find the infinite-horizon cost-to-go matrix P
    P_dare = solve_discrete_are(A, B, np.array(Q), np.array(R))
    
    # 6. Regularize P_dare and compute its Cholesky decomposition
    L = np.linalg.cholesky(P_dare - reg * np.eye(n_x))
    
    # 7. Extract lower-triangular elements and convert back to JAX array
    c_q = L[np.tril_indices(n_x)]
    
    return jnp.array(c_q)