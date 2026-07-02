"""Autonomous car (dynamic bicycle model) environment for bpmpc."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Mapping, Tuple
from scipy.interpolate import splprep, splev, interp1d
from scipy.integrate import cumulative_trapezoid

from bpmpc.dynamics import Dynamics
from bpmpc.env._src.integrators import rk4_integrator


class LinearizedAutonomousCar:
    """Dynamic bicycle model of an autonomous car.
    
    This system models the lateral vehicle dynamics assuming a constant 
    longitudinal velocity and linear tire characteristics. It includes a 
    utility method to generate reference waypoints from a set of track points.

    State Vector (x) - Shape (4,):
        x[0]: Lateral position error / global y [m]
        x[1]: Lateral velocity (v_y) [m/s]
        x[2]: Yaw angle (psi) [rad]
        x[3]: Yaw rate (r) [rad/s]
        
    Control Input (u) - Shape (1,):
        u[0]: Steering angle (delta) [rad]

    Parameters
    ----------
    dt : float, optional
        The discrete-time integration step size in seconds, by default 0.01.
    velocity : float, optional
        The constant longitudinal velocity of the car in m/s, by default 5.0.
    """

    def __init__(self, dt: float = 0.01, velocity: float = 5.0):
        self.dt = dt
        self.v_x = velocity
        
        # Physical Parameters
        self.c_f = 155494.663
        self.c_r = 155494.663
        self.m = 1140.0
        self.l_f = 1.165
        self.l_r = 1.165
        self.i_z = 1436.24

        # Continuous-time Nominal Matrices (A and B)
        self.A_nom = jnp.array([
            [0.0, 1.0, self.v_x, 0.0],
            [0.0, -(self.c_f + self.c_r) / (self.m * self.v_x), 0.0, (self.c_r * self.l_r - self.c_f * self.l_f) / (self.m * self.v_x) - self.v_x],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, (self.c_r * self.l_r - self.c_f * self.l_f) / (self.i_z * self.v_x), 0.0, -(self.c_f * self.l_f**2 + self.c_r * self.l_r**2) / (self.i_z * self.v_x)]
        ])

        self.B_nom = jnp.array([
            [0.0],
            [self.c_f / self.m],
            [0.0],
            [self.c_f * self.l_f / self.i_z]
        ])

        # Disturbance mapping matrix (affects lateral vel and yaw rate)
        self.G_mat = jnp.array([
            [0.0],
            [1.0],
            [0.0],
            [1.0]
        ])

    def ode(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Evaluates the continuous-time linear parameter-varying ODEs.

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (4,).
        u : jax.Array
            Current control input, shape (1,).
        params : Mapping[str, jax.Array]
            Dictionary containing optional parametric uncertainties:
            - `theta`: Array of shape (8,) containing the unknown system parameters
                       used for System Identification. Only elements 0-5 are 
                       active, mapping directly to perturbations in lateral velocity 
                       and yaw rate derivatives.
        """
        dx = self.A_nom @ x + self.B_nom @ u

        # Inject specific parametric uncertainty matrix structure from sys_id
        if "theta" in params:
            theta = params["theta"]
            
            # Row 1 perturbation (lateral velocity derivative)
            d_dx1 = theta[0]*x[1] + theta[1]*x[3] + theta[2]*u[0]
            # Row 3 perturbation (yaw rate derivative)
            d_dx3 = theta[3]*x[1] + theta[4]*x[3] + theta[5]*u[0]
            
            dx = dx.at[1].add(d_dx1)
            dx = dx.at[3].add(d_dx3)

        return dx

    def step(self, x: jax.Array, u: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Advances the system one discrete time step via RK4 integration.

        Parameters
        ----------
        x : jax.Array
            Current state vector, shape (4,).
        u : jax.Array
            Current control input, shape (1,).
        params : Mapping[str, jax.Array]
            Dictionary containing system parameters. In addition to `theta`, it accepts:
            - `w`: Additive disturbance vector, shape (1,). Maps through `self.G_mat`.
        """
        # Note: Depending on your exact preference, you can swap this to Forward Euler:
        # x_next = x + self.ode(x, u, params) * self.dt
        x_next = rk4_integrator(self.ode, x, u, params, self.dt)
        
        # Add additive disturbance mapped to specific channels
        if "w" in params:
            w = params["w"]
            x_next += (self.G_mat @ w) * self.dt
            
        return x_next

    def get_dynamics(self) -> Dynamics:
        """Wraps the environment in a `bpmpc.dynamics.Dynamics` object."""
        return Dynamics(
            true_fun=self.step,
            nominal_fun=self.step
        )

    # ======================================================================
    # CPU / NumPy Methods for Track Generation
    # ======================================================================

    def generate_waypoints(self, track: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates uniformly sampled reference waypoints along a spline track.

        This method operates entirely in standard Python/NumPy (not JAX) and 
        should be called once before the simulation to construct the trajectory.

        Parameters
        ----------
        track : np.ndarray
            An array of shape (N, 2) containing the [x, y] coordinates of the 
            track vertices.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - x_samples: X-coordinates of the waypoints.
            - y_samples: Y-coordinates of the waypoints.
            - theta_samples: Tangent angle (yaw) of the path at each waypoint.
        """
        # Create spline representation
        x_track, y_track = track[:, 0], track[:, 1]
        tck, _ = splprep([x_track, y_track], s=0, k=2, per=0)

        # Dense parameter grid to compute arc length along the spline
        u_dense = np.linspace(0.0, 1.0, 1_000_000)
        dx_du, dy_du = splev(u_dense, tck, der=1)
        speed_u = np.hypot(dx_du, dy_du)           # ds/du

        # s(u): cumulative arc length along the spline
        s_of_u = np.concatenate(([0.0], cumulative_trapezoid(speed_u, u_dense)))
        total_length = s_of_u[-1]

        # Make s strictly increasing (handles flat spots)
        mask = np.r_[True, np.diff(s_of_u) > 1e-12]
        u_mon = u_dense[mask]
        s_mon = s_of_u[mask]

        # Invert: u(s)
        u_of_s = interp1d(s_mon, u_mon, kind='linear', assume_sorted=True)

        # Sample at constant speed v with sampling time dt
        ds = self.v_x * self.dt
        s_samples = np.arange(0.0, total_length + 1e-12, ds)

        # Compute positions
        u_samples = u_of_s(s_samples)
        x_samples, y_samples = splev(u_samples, tck)

        # Compute headings (yaw angles)
        dx_du_samples, dy_du_samples = splev(u_samples, tck, der=1)
        theta_samples = np.unwrap(np.arctan2(dy_du_samples, dx_du_samples))

        return x_samples, y_samples, theta_samples