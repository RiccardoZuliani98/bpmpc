"""Nonlinear cart-pendulum environment for bpmpc_jax."""

import jax
import jax.numpy as jnp
from typing import Mapping

from bpmpc.dynamics import Dynamics
from bpmpc.variable import Variable
from bpmpc.env._src.integrators import rk4_integrator


class CartPendulum:
    """Nonlinear cart-pendulum (cart-pole) dynamical system.
    
    This class models a cart with mass `m` moving on a frictionless track, 
    carrying a pendulum with inertia `j` and coupling mass `mu`. The system 
    is defined in continuous time and discretized using RK4 integration.

    State Vector (x) - Shape (4,):
        x[0] (p):         Cart position [m]
        x[1] (p_dot):     Cart velocity [m/s]
        x[2] (theta):     Pendulum angle [rad] (0 rad is pointing straight up)
        x[3] (theta_dot): Pendulum angular velocity [rad/s]

    Control Input (u) - Shape (1,):
        u[0] (F):         Force applied to the cart [N]

    Parameters
    ----------
    dt : float, optional
        The discrete-time integration step size in seconds, by default 0.05.
    """
    
    def __init__(self, dt: float = 0.05):
        self.dt = dt
        
        # Base nominal physical parameters
        self.m_nom = 0.305      # Total mass
        self.mu_nom = 1.5492e-3 # Coupling mass
        self.j_nom = 1.47e-4    # Pendulum inertia
        self.g = 9.804          # Gravity

    def ode(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Evaluates the continuous-time ordinary differential equations (ODEs).

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (4,).
        u : jax.Array
            Current control input, shape (1,).
        params : Mapping[str, jax.Array]
            Dictionary containing optional parametric uncertainties:
            - `d`: Array of shape (2,) representing relative perturbations to total mass 
              and coupling mass respectively (e.g., 0.1 for +10%). Default: [0.0, 0.0].

        Returns
        -------
        jax.Array
            The state derivative vector [p_dot, p_ddot, theta_dot, theta_ddot], shape (4,).
        """
        p_dot, theta_dot = x[1], x[3]
        theta = x[2]
        F = u[0]
        
        # Inject parametric uncertainties if provided
        d = params.get("d", jnp.zeros(2))
        d_m = d[0]
        d_mu = d[1]
        
        m = self.m_nom * (1.0 + d_m)
        mu = self.mu_nom * (1.0 + d_mu)
        j_val = self.j_nom
        
        denom = m * j_val - mu**2 * jnp.cos(theta)**2
        force_term = F + mu * theta_dot**2 * jnp.sin(theta)
        
        p_ddot = (j_val * force_term - mu**2 * self.g * jnp.sin(theta) * jnp.cos(theta)) / denom
        theta_ddot = (m * mu * self.g * jnp.sin(theta) - mu * jnp.cos(theta) * force_term) / denom
        
        return jnp.array([p_dot, p_ddot, theta_dot, theta_ddot])

    def step(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Advances the system one discrete time step via RK4 integration.

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (4,).
        u : jax.Array
            Current control input, shape (1,).
        params : Mapping[str, jax.Array]
            Dictionary containing system parameters. In addition to the physics 
            parameters used in `ode()`, this function accepts:
            - `w`: Additive state disturbance (process noise) vector, shape (4,). Default: zeros.

        Returns
        -------
        jax.Array
            The next state vector, shape (4,).
        """
        x_next = rk4_integrator(self.ode, x, u, params, self.dt)
        w = params.get("w", jnp.zeros_like(x))
        return x_next + w

    def get_dynamics(self, include_w: bool = False, include_d: bool = False) -> Dynamics:
        """Wraps the environment in a `bpmpc_jax.dynamics.Dynamics` object.

        This enables the straight-through gradient estimator pattern, allowing
        the true simulation pass and the nominal autodiff pass to evaluate 
        different parametric dictionaries.

        Parameters
        ----------
        include_w : bool, optional
            If True, enables additive state disturbance (process noise) via the "w" key.
            Expects a shape of (4,).
        include_d : bool, optional
            If True, enables parametric model uncertainty via the "d" key.
            Expects a shape of (2,).

        Returns
        -------
        Dynamics
            The JAX-differentiable dynamics wrapper.
        """
        
        # 1. Dynamically build the parameter specification
        true_spec = []
        if include_w:
            true_spec.append(Variable("w", shape=(4,)))
        if include_d:
            true_spec.append(Variable("d", shape=(2,)))
            
        def true_fun(x, u, params):
            active_params = {}
            if include_w and "w" in params:
                active_params["w"] = params["w"]
            if include_d and "d" in params:
                active_params["d"] = params["d"]
            return self.step(x, u, active_params)
            
        def nominal_fun(x, u, params):
            return self.step(x, u, {})

        return Dynamics(
            true_fun=true_fun,
            nominal_fun=nominal_fun,
            nx=4,
            nu=1,
            true_params_spec=tuple(true_spec)  # 2. Pass the spec to the constructor
        )