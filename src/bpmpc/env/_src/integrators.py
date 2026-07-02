import jax
import jax.numpy as jnp
from typing import Mapping

def rk4_integrator(ode_fn, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array], dt: float) -> jax.Array:
    """Standard Runge-Kutta 4 discrete-time integrator with additive noise."""
    k1 = ode_fn(x, u, params)
    k2 = ode_fn(x + (dt / 2.0) * k1, u, params)
    k3 = ode_fn(x + (dt / 2.0) * k2, u, params)
    k4 = ode_fn(x + dt * k3, u, params)
    
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    # Add additive state disturbance (process noise) if provided in `params`
    w = params.get("w", jnp.zeros_like(x))
    return x_next + w