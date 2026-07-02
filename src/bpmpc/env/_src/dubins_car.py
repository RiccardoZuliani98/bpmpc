"""Dubins car environment for bpmpc."""

import jax
import jax.numpy as jnp
from typing import Mapping

from bpmpc.dynamics import Dynamics


class DubinsCar:
    """Discrete-time Dubins car dynamical system.
    
    This system models a standard Dubins car using a 3-dimensional state 
    space and a 2-dimensional control space. It supports optional additive 
    process noise `w` on the translational positions.

    State Vector (x) - Shape (3,):
        x[0]: p_x (X position) [m]
        x[1]: p_y (Y position) [m]
        x[2]: theta (Heading angle) [rad]
        
    Control Input (u) - Shape (2,):
        u[0]: v (Forward velocity) [m/s]
        u[1]: omega (Angular velocity) [rad/s]

    Parameters
    ----------
    dt : float, optional
        The discrete-time Forward Euler integration step size in seconds, by default 0.1.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.n_x = 3
        self.n_u = 2
        
        # Default reference input (zero velocity, zero turning)
        self.u_ref = jnp.zeros(self.n_u)

    def ode(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Evaluates the continuous-time nonlinear ODEs of the Dubins car.

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (3,).
        u : jax.Array
            Current control input, shape (2,).
        params : Mapping[str, jax.Array]
            Dictionary containing optional system parameters (unused for nominal ODE, 
            but kept for API compatibility with bpmpc).
        """
        # Unpack states and controls
        p_x, p_y, theta = x
        v, omega = u
        
        # Dubins car kinematic equations
        p_x_dot = v * jnp.cos(theta)
        p_y_dot = v * jnp.sin(theta)
        theta_dot = omega
        
        return jnp.array([
            p_x_dot, 
            p_y_dot, 
            theta_dot
        ])

    def step(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Advances the system one discrete time step via Forward Euler.

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (3,).
        u : jax.Array
            Current control input, shape (2,).
        params : Mapping[str, jax.Array]
            Dictionary containing system parameters. Accepts:
            - `w`: Additive disturbance vector acting on the positions (p_x, p_y).
                   Shape (2,). Maps directly into the first 2 elements of the next state.
        """
        # Forward Euler Step
        x_next = x + self.dt * self.ode(x, u, params)
        
        # Add additive disturbance mapped to position channels (x, y)
        if "w" in params:
            # Pad the 2-element noise vector with 1 zero for the heading (theta) state
            w_full = jnp.pad(params["w"], (0, 1))
            x_next += w_full
            
        return x_next

    def get_dynamics(self) -> Dynamics:
        """Wraps the environment in a `bpmpc_jax.dynamics.Dynamics` object."""
        return Dynamics(
            true_fun=self.step,
            nominal_fun=self.step,
            nx=self.n_x,
            nu=self.n_u
        )