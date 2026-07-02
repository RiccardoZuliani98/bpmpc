"""Randomly generated discrete-time parameterized linear system environment."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm
from typing import Mapping, Tuple

from bpmpc.dynamics import Dynamics

class ParameterizedLinear:
    """Parameterized discrete-time linear system.
    
    This class defines a generic controllable linear system environment 
    where the system matrices are strictly driven by the parameter vector `theta`.
    
    State Vector (x) - Shape (n_x,):
        x: The linear state vector.
        
    Control Input (u) - Shape (1,):
        u: The scalar control input.

    Parameters
    ----------
    n_x : int, optional
        The number of states in the system, by default 2.
    """

    def __init__(
        self, 
        n_x: int = 2
    ):
        self.n_x = n_x
        self.n_u = 1

    def generate_theta(
        self, 
        key: jax.Array, 
        pole_range: Tuple[float, float] = (0.5, 1.2),
        dt: float = 0.1
    ) -> jax.Array:
        """Generates a random parameter vector (flattened A and B) for the system.
        
        Samples continuous-time poles uniformly, constructs the system in 
        controllable canonical form, and applies an exact Zero-Order Hold (ZOH).
        
        Note: Because this uses `scipy.linalg.expm` and standard `numpy`, 
        this method should be called outside of JIT-compiled functions.

        Parameters
        ----------
        key : jax.Array
            A JAX PRNGKey used to seed the random pole generation.
        pole_range : Tuple[float, float], optional
            The [min, max] range to uniformly sample the real continuous-time 
            poles. Positive values create unstable open-loop systems.
        dt : float, optional
            The discrete-time sampling step size in seconds, by default 0.1.

        Returns
        -------
        jax.Array
            A 1D array of shape `(n_x * n_x + n_x,)` containing the flattened 
            A matrix followed by the flattened B matrix.
        """
        # 1. Generate random continuous-time poles using JAX PRNG
        poles = jax.random.uniform(key, shape=(self.n_x,), minval=pole_range[0], maxval=pole_range[1])
        
        # 2. Construct continuous-time system in controllable canonical form
        A_cont = jnp.diag(jnp.ones(self.n_x - 1), k=1)
        A_cont = A_cont.at[-1, :].set(-jnp.flip(jnp.poly(poles)[1:]))
        
        B_cont = jnp.zeros((self.n_x, 1))
        B_cont = B_cont.at[-1, 0].set(1.0)

        # 3. Exact Zero-Order Hold (ZOH) Discretization using matrix exponential
        M = jnp.zeros((self.n_x + 1, self.n_x + 1))
        M = M.at[:self.n_x, :self.n_x].set(A_cont)
        M = M.at[:self.n_x, self.n_x:].set(B_cont)

        M_exp = expm(M*dt)
        
        # Extract discrete matrices and flatten into theta
        A = M_exp[:self.n_x, :self.n_x]
        B = M_exp[:self.n_x, self.n_x:]
        
        return jnp.concatenate([A.flatten(), B.flatten()])

    def theta_to_matrix(self, theta: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Reconstructs the system matrices A and B from the flattened parameter vector.
        
        Parameters
        ----------
        theta : jax.Array
            A 1D array of shape `(n_x * n_x + n_x,)` containing the flattened A and B matrices.
            
        Returns
        -------
        tuple[jax.Array, jax.Array]
            A tuple containing:
            - A: The reconstructed A matrix of shape `(n_x, n_x)`.
            - B: The reconstructed B matrix of shape `(n_x, n_u)`.
        """
        split_idx = self.n_x * self.n_x
        
        A_flat = theta[:split_idx]
        B_flat = theta[split_idx:]
        
        A = A_flat.reshape((self.n_x, self.n_x))
        B = B_flat.reshape((self.n_x, self.n_u)) 
        
        return A, B

    def step(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Advances the parameterized linear system one discrete time step.

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (n_x,).
        u : jax.Array
            Current control input, shape (1,).
        params : Mapping[str, jax.Array]
            Dictionary defining the system parameters and uncertainties:
            - `theta` (Required): 1D array of shape `(n_x * n_x + n_x,)` containing
                                  the flattened system matrices A and B.
            - `w` (Optional): Additive state disturbance (process noise), shape (n_x,).

        Returns
        -------
        jax.Array
            The next state vector, shape (n_x,).
        """
        if "theta" not in params:
            raise ValueError("The 'params' mapping must contain 'theta' to step the ParameterizedLinear system.")
            
        A, B = self.theta_to_matrix(params["theta"])

        x_next = A @ x + B @ u
        
        # Add additive disturbance mapped to all channels
        if "w" in params:
            x_next += params["w"]
            
        return x_next

    def get_dynamics(self) -> Dynamics:
        """Wraps the environment in a `bpmpc_jax.dynamics.Dynamics` object."""
        return Dynamics(
            true_fun=self.step,
            nominal_fun=self.step,
            nx=self.n_x,
            nu=self.n_u
        )