"""Linearized aircraft environment for bpmpc."""

import jax
import jax.numpy as jnp
from typing import Mapping

from bpmpc.dynamics import Dynamics


class LinearizedAircraft:
    """Discrete-time linearized aircraft longitudinal dynamics.
    
    This system models the simplified discrete-time pitch dynamics of an 
    aircraft. The model is taken from "Reference tracking MPC using dynamic 
    terminal set transformation" by Daniel Simon et al. (IEEE TAC, 2014).

    State Vector (x) - Shape (2,):
        x[0]: Pitch angle [rad]
        x[1]: Pitch rate [rad/s]
        
    Control Input (u) - Shape (1,):
        u[0]: Elevator deflection [rad]
    """
    
    def __init__(self):
        # Nominal system matrix (A)
        self.A_nom = jnp.array([
            [0.9719, 0.0155],
            [0.2097, 0.9705]
        ])

        # Nominal input matrix (B)
        self.B_nom = jnp.array([
            [0.0071],
            [0.3263]
        ])

    def step(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Advances the linear system one discrete time step.

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (2,).
        u : jax.Array
            Current control input, shape (1,).
        params : Mapping[str, jax.Array]
            Dictionary containing optional additive uncertainties:
            - `d_A`: Matrix perturbation added to A_nom, shape (2, 2). Default: zeros.
            - `d_B`: Matrix perturbation added to B_nom, shape (2, 1). Default: zeros.
            - `w`: Additive state disturbance (process noise), shape (2,). Default: zeros.

        Returns
        -------
        jax.Array
            The next state vector, shape (2,).
        """
        A = self.A_nom + params.get("d_A", jnp.zeros_like(self.A_nom))
        B = self.B_nom + params.get("d_B", jnp.zeros_like(self.B_nom))
        
        x_next = A @ x + B @ u
        w = params.get("w", jnp.zeros_like(x))
        
        return x_next + w

    def get_dynamics(self) -> Dynamics:
        """Wraps the environment in a `bpmpc_jax.dynamics.Dynamics` object.

        This enables the straight-through gradient estimator pattern, allowing
        the true simulation pass and the nominal autodiff pass to evaluate 
        different parametric dictionaries.

        Returns
        -------
        Dynamics
            The JAX-differentiable dynamics wrapper.
        """
        return Dynamics(
            true_fun=self.step,
            nominal_fun=self.step
        )