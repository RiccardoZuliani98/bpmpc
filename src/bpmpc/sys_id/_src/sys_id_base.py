"""System Identification module for adaptive Model Predictive Control (MPC).

This module provides tools to automatically formulate and solve Recursive 
Least Squares (RLS) estimation problems for parameterized dynamical systems. 
It leverages JAX's forward-mode auto-differentiation to extract affine features 
from non-linear dynamics without requiring manual analytical derivations.
"""

import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Tuple, Optional

class SysIDState(NamedTuple):
    """Holds the recursive state for the system identifier.
    
    Attributes
    ----------
    theta : jax.Array
        The current parameter estimate vector of shape `(n_theta,)`.
    P_inv : jax.Array
        The inverse covariance matrix (Information matrix) of shape `(n_theta, n_theta)`.
    """
    theta: jax.Array
    P_inv: jax.Array


def extract_features(
    f: Callable, 
    n_theta: int,
    verify: bool = False,
    x_probe: Optional[jax.Array] = None,
    u_probe: Optional[jax.Array] = None,
) -> Tuple[Callable, Callable]:
    """Automatically extracts the nominal drift `g` and feature matrix `psi` 
    from a generic parameter-affine dynamics function.
    
    Assuming the user's dynamics function has the form:
        f(x, u, theta) = g(x, u) + psi(x, u) @ theta
        
    This function uses `jax.jacfwd` to analytically extract `g` (by evaluating 
    at theta=0) and `psi` (by taking the Jacobian w.r.t theta).

    Parameters
    ----------
    f : Callable
        The true system dynamics function with signature `f(x, u, theta, params)`.
    n_theta : int
        The dimensionality of the unknown parameter vector `theta`.
    verify : bool, optional
        If True, verifies that the function is strictly affine in `theta` by 
        evaluating a random test point. Defaults to False.
    x_probe : jax.Array, optional
        A dummy state array used for the verification check. Required if `verify=True`.
    u_probe : jax.Array, optional
        A dummy control array used for the verification check. Required if `verify=True`.

    Returns
    -------
    Tuple[Callable, Callable]
        - g(x, u, params): Function returning the nominal drift (shape `nx`).
        - psi(x, u, params): Function returning the feature matrix (shape `(nx, n_theta)`).
        
    Raises
    ------
    ValueError
        If `verify=True` but probe arrays are not provided, or if the provided 
        dynamics function `f` is found to be non-affine in `theta`.
    """
    def g(x: jax.Array, u: jax.Array, params: Optional[dict] = None) -> jax.Array:
        # The nominal drift is simply the dynamics evaluated with theta = 0
        return f(x, u, jnp.zeros(n_theta), params or {})

    def psi(x: jax.Array, u: jax.Array, params: Optional[dict] = None) -> jax.Array:
        # The feature matrix is the Jacobian of f with respect to theta at theta = 0
        return jax.jacfwd(f, argnums=2)(x, u, jnp.zeros(n_theta), params or {})

    if verify:
        if x_probe is None or u_probe is None:
            raise ValueError("x_probe and u_probe must be provided if verify=True.")
        
        # 1. Generate a random parameter vector for testing
        key = jax.random.PRNGKey(42)
        test_theta = jax.random.normal(key, (n_theta,))
        
        # 2. Evaluate the user's true function
        f_val = f(x_probe, u_probe, test_theta, {})
        
        # 3. Evaluate our extracted feature formulation
        g_val = g(x_probe, u_probe, {})
        psi_val = psi(x_probe, u_probe, {})
        affine_val = g_val + psi_val @ test_theta
        
        # 4. Enforce strict numerical equality
        if not jnp.allclose(f_val, affine_val, rtol=1e-5, atol=1e-5):
            max_err = jnp.max(jnp.abs(f_val - affine_val))
            raise ValueError(
                f"Dynamics function is not affine in theta! "
                f"Max deviation from g(x) + psi(x)*theta: {max_err:.3e}\n"
                f"Ensure the parameters enter the dynamics strictly linearly."
            )

    return g, psi


class RLS:
    """JAX-compatible Recursive Least Squares estimator.
    
    This class tracks and updates the parameter estimate `theta` using an 
    Information Filter formulation. By propagating the inverse covariance matrix 
    (`P_inv`), it avoids matrix inversion lemmas and naturally supports batched 
    (block) updates over trajectory segments.
    """
    
    def __init__(self, n_theta: int, g_fn: Callable, psi_fn: Callable, lam: float = 1.0):
        """
        Parameters
        ----------
        n_theta : int
            The dimensionality of the parameter vector `theta`.
        g_fn : Callable
            The nominal drift function `g(x, u)` extracted from the dynamics.
        psi_fn : Callable
            The feature matrix function `psi(x, u)` extracted from the dynamics.
        lam : float, optional
            The forgetting factor in (0, 1]. A value of 1.0 implies infinite memory, 
            while values < 1.0 exponentially discount past data. Defaults to 1.0.
        """
        self.n_theta = n_theta
        self.g_fn = g_fn
        self.psi_fn = psi_fn
        self.lam = lam

    def init(self, theta0: jax.Array, P0_inv: jax.Array) -> SysIDState:
        """Initializes the RLS state.
        
        Parameters
        ----------
        theta0 : jax.Array
            The initial guess for the parameters, shape `(n_theta,)`.
        P0_inv : jax.Array
            The initial Information matrix (inverse covariance), shape `(n_theta, n_theta)`.
            Higher values indicate higher confidence in `theta0`.
            
        Returns
        -------
        SysIDState
            The initialized estimator state.
        """
        return SysIDState(theta=theta0, P_inv=P0_inv)

    def step(
        self, 
        state: SysIDState, 
        x: jax.Array, 
        u: jax.Array, 
        x_next: jax.Array
    ) -> SysIDState:
        """Performs a Recursive Least Squares update.
        
        This method supports both sequential (1D) vectors and batched (2D) matrices.
        If a batch of shape `(B, nx)` is provided, it performs a "Block RLS" update, 
        treating all samples in the batch as arriving simultaneously and applying 
        the forgetting factor `lam` exactly once.

        Parameters
        ----------
        state : SysIDState
            The current state of the estimator (`theta` and `P_inv`).
        x : jax.Array
            The state at time k. Can be shape `(nx,)` or a batch `(B, nx)`.
        u : jax.Array
            The control input at time k. Can be shape `(nu,)` or a batch `(B, nu)`.
        x_next : jax.Array
            The observed state at time k+1. Can be shape `(nx,)` or a batch `(B, nx)`.

        Returns
        -------
        SysIDState
            The updated RLS state.
        """
        
        # --- BATCHED INPUTS (B, nx) ---
        if x.ndim == 2:
            # Safely evaluate features for each item in the batch
            g_val = jax.vmap(self.g_fn)(x, u)       # shape: (B, nx)
            psi_val = jax.vmap(self.psi_fn)(x, u)   # shape: (B, nx, n_theta)
            
            # Compute measurement mismatch
            y = x_next - g_val                      # shape: (B, nx)
            
            # Flatten the batch and state dimensions into tall 2D matrices
            psi_tall = psi_val.reshape(-1, self.n_theta) # shape: (B * nx, n_theta)
            y_tall = y.reshape(-1)                       # shape: (B * nx,)
            
            # Aggregate information over the block
            info_update = psi_tall.T @ psi_tall
            rhs_update = psi_tall.T @ y_tall
            
        # --- SINGLE INPUTS (nx,) ---
        else:
            g_val = self.g_fn(x, u)                 # shape: (nx,)
            psi_val = self.psi_fn(x, u)             # shape: (nx, n_theta)
            
            y = x_next - g_val                      # shape: (nx,)
            
            info_update = psi_val.T @ psi_val
            rhs_update = psi_val.T @ y

        # Update the Information matrix: P_inv_next = lam * P_inv + psi^T * psi
        P_inv_next = self.lam * state.P_inv + info_update
        
        # Formulate the RHS for the theta update
        rhs = self.lam * (state.P_inv @ state.theta) + rhs_update
        
        # Solve for the new parameter estimate using Cholesky factorization 
        # optimized for symmetric positive-definite matrices
        theta_next = jax.scipy.linalg.solve(P_inv_next, rhs, assume_a='pos')
        
        return SysIDState(theta=theta_next, P_inv=P_inv_next)