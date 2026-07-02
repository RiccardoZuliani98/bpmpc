"""Quadcopter environment for bpmpc."""

import jax
import jax.numpy as jnp
from typing import Mapping

from bpmpc.dynamics import Dynamics


class Quadcopter:
    """Discrete-time quadcopter dynamical system.
    
    This system models a standard quadcopter using a 12-dimensional state 
    space and a 4-dimensional control space (rotor speeds). It natively supports 
    a 12-dimensional parameter uncertainty vector `theta` for system identification, 
    and additive process noise `w` on the translational positions.

    State Vector (x) - Shape (12,):
        x[0]: p_x (Inertial X position) [m]
        x[1]: p_y (Inertial Y position) [m]
        x[2]: p_z (Inertial Z position) [m]
        x[3]: v_x (Inertial X velocity) [m/s]
        x[4]: v_y (Inertial Y velocity) [m/s]
        x[5]: v_z (Inertial Z velocity) [m/s]
        x[6]: roll (Euler angle φ) [rad]
        x[7]: pitch (Euler angle θ) [rad]
        x[8]: yaw (Euler angle ψ) [rad]
        x[9]: p (Body roll rate) [rad/s]
        x[10]: q (Body pitch rate) [rad/s]
        x[11]: r (Body yaw rate) [rad/s]
        
    Control Input (u) - Shape (4,):
        u[0]: omega_1 (Rotor 1 speed) [rad/s]
        u[1]: omega_2 (Rotor 2 speed) [rad/s]
        u[2]: omega_3 (Rotor 3 speed) [rad/s]
        u[3]: omega_4 (Rotor 4 speed) [rad/s]

    Parameters
    ----------
    dt : float, optional
        The discrete-time Forward Euler integration step size in seconds, by default 0.1.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        
        # True system physical constants
        self.mass = 0.468
        self.length = 0.225
        self.j_r = 3.357e-5
        self.k_t = 2.98e-6
        self.k_b = 1.14e-7
        self.i_xx = 4.856e-3
        self.i_yy = 4.856e-3
        self.i_zz = 8.801e-3
        self.k_dx = 0.25
        self.k_dy = 0.25
        self.k_dz = 0.25
        self.gravity = 9.81
        self.n_x = 12
        self.n_u = 4
        self.u_ref = jnp.ones(self.n_u) * jnp.sqrt((self.mass*self.gravity)/(4*self.k_t))  # each rotor at hover

    def get_theta_true(self) -> jax.Array:
        """Returns the nominal parameter vector `theta` based on true physical constants.
        
        This 12D vector represents the exact system formulation used by the 
        nominal feature extractor / estimator.
        """
        return jnp.array([
            self.k_t / self.mass,
            self.k_dx / self.mass,
            self.k_dy / self.mass,
            self.k_dz / self.mass,
            (self.i_yy - self.i_zz) / self.i_xx,
            self.j_r / self.i_xx,
            self.length * self.k_t / self.i_xx,
            (self.i_zz - self.i_xx) / self.i_yy,
            self.j_r / self.i_yy,
            self.length * self.k_t / self.i_yy,
            (self.i_xx - self.i_yy) / self.i_zz,
            self.k_b / self.i_zz
        ])

    def get_hover_thrust(self) -> jax.Array:
        """Returns the baseline rotor speeds `u_ref` required to maintain hover."""
        omega_hover = jnp.sqrt(self.mass * self.gravity / (4.0 * self.k_t))
        return jnp.full((4,), omega_hover)

    def ode(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Evaluates the continuous-time nonlinear ODEs of the quadcopter.

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (12,).
        u : jax.Array
            Current control input, shape (4,).
        params : Mapping[str, jax.Array]
            Dictionary containing optional parametric uncertainties:
            - `theta`: Array of shape (12,) overriding the true physical parameters.
                       If not provided, defaults to `get_theta_true()`.
        """
        p_x, p_y, p_z, p_x_dot, p_y_dot, p_z_dot, roll, pitch, yaw, p, q, r = x
        omega_1, omega_2, omega_3, omega_4 = u
        
        # Use provided theta for SysID, otherwise fall back to exact physics
        theta = params.get("theta", self.get_theta_true())
        
        omega_r = omega_2 - omega_1 + omega_4 - omega_3
        
        # Compute angular accelerations in body frame
        p_dot = theta[4] * q * r - theta[5] * q * omega_r + theta[6] * (omega_4**2 - omega_2**2)
        q_dot = theta[7] * p * r + theta[8] * p * omega_r + theta[9] * (omega_3**2 - omega_1**2)
        r_dot = theta[10] * p * q + theta[11] * (omega_4**2 - omega_3**2 + omega_2**2 - omega_1**2)
        
        # Compute linear accelerations in inertial frame
        thrust_term = theta[0] * (omega_1**2 + omega_2**2 + omega_3**2 + omega_4**2)
        
        p_x_ddot = thrust_term * (jnp.cos(yaw) * jnp.sin(pitch) * jnp.cos(roll) + jnp.sin(yaw) * jnp.sin(roll)) - theta[1] * p_x_dot
        p_y_ddot = thrust_term * (jnp.sin(yaw) * jnp.sin(pitch) * jnp.cos(roll) - jnp.cos(yaw) * jnp.sin(roll)) - theta[2] * p_y_dot
        p_z_ddot = -self.gravity + thrust_term * jnp.cos(pitch) * jnp.cos(roll) - theta[3] * p_z_dot
        
        # Compute Euler angle derivatives using T(roll, pitch) matrix
        roll_dot = p + jnp.sin(roll) * jnp.tan(pitch) * q + jnp.cos(roll) * jnp.tan(pitch) * r
        pitch_dot = jnp.cos(roll) * q - jnp.sin(roll) * r
        yaw_dot = (jnp.sin(roll) / jnp.cos(pitch)) * q + (jnp.cos(roll) / jnp.cos(pitch)) * r
        
        return jnp.array([
            p_x_dot, p_y_dot, p_z_dot,
            p_x_ddot, p_y_ddot, p_z_ddot,
            roll_dot, pitch_dot, yaw_dot,
            p_dot, q_dot, r_dot
        ])

    def step(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Advances the system one discrete time step via Forward Euler.

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (12,).
        u : jax.Array
            Current control input, shape (4,).
        params : Mapping[str, jax.Array]
            Dictionary containing system parameters. In addition to `theta`, it accepts:
            - `w`: Additive disturbance vector acting on the positions (p_x, p_y, p_z).
                   Shape (3,). Maps directly into the first 3 elements of the next state.
        """
        # Forward Euler Step
        x_next = x + self.dt * self.ode(x, u, params)
        
        # Add additive disturbance mapped to position channels
        if "w" in params:
            # Pad the 3-element noise vector with 9 zeros for the rest of the states
            w_full = jnp.pad(params["w"], (0, 9))
            x_next += w_full
            
        return x_next

    def get_dynamics(self) -> Dynamics:
        """Wraps the environment in a `bpmpc.dynamics.Dynamics` object."""
        return Dynamics(
            true_fun=self.step,
            nominal_fun=self.step,
            nx=self.n_x,
            nu=self.n_u
        )