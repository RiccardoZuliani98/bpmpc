import pytest
import jax
import jax.numpy as jnp
import numpy as np

from bpmpc.mpc import Cost, Constraint, MPCProblem
from bpmpc.mpc.helpers import lti_dynamics, box_bounds
from bpmpc.variable import Variable
from bpmpc.dynamics import Dynamics
from bpmpc.closed_loop import ClosedLoop
from jaxsparrow import setup_dense_solver

jax.config.update("jax_enable_x64", True)


# ======================================================================
# Helper: Vmap over JVP to compute Jacobians without jacfwd
# ======================================================================
def _compute_jacobian(f, x):
    """Computes the Jacobian using vmap over jax.jvp to avoid jax.jacfwd."""
    I = jnp.eye(x.shape[0])
    
    def pushfwd(v):
        # jvp returns (primal_out, tangent_out). We only want the tangent.
        return jax.jvp(f, (x,), (v,))[1]
    
    # vmap over the tangent vectors (rows of I)
    # Output shape will be (nx, *f(x).shape)
    J_vmap = jax.vmap(pushfwd)(I)
    
    # Standard Jacobian shape is (*f(x).shape, nx), so we move the batched axis to the end
    return jnp.moveaxis(J_vmap, 0, -1)


# ======================================================================
# Helper: Build an Unconstrained LQR-like MPC Problem
# ======================================================================
def setup_lqr_mpc(nx=2, nu=1, N=10):
    """Creates a simple MPC with loose bounds so it behaves purely linearly."""
    v_x0 = Variable("x0", (nx,))
    v_A = Variable("A", (nx, nx))
    v_B = Variable("B", (nx, nu))
    v_Q = Variable("Q", (nx, nx))
    v_R = Variable("R", (nu, nu))

    mpc_dyn = lti_dynamics(A=v_A, B=v_B, x0=v_x0, horizon=N)

    mpc_bounds = box_bounds(
        x_min=jnp.full((nx,), -1000.0), x_max=jnp.full((nx,), 1000.0),
        u_min=jnp.full((nu,), -1000.0), u_max=jnp.full((nu,), 1000.0),
        horizon=N
    )

    def q_mat(v):
        Q_blk = jnp.kron(jnp.eye(N), v["Q"])
        R_blk = jnp.kron(jnp.eye(N), v["R"])
        return jnp.block([
            [Q_blk, jnp.zeros((N * nx, N * nu))],
            [jnp.zeros((N * nu, N * nx)), R_blk],
        ])

    mpc_cost = Cost(q_mat=q_mat, v_in_q_mat={"Q": v_Q, "R": v_R})

    problem = MPCProblem(
        costs=[mpc_cost],
        constraints=[mpc_dyn, mpc_bounds],
        slow_vars=["A", "B", "Q", "R"],
        outputs={"u0": slice(N * nx, N * nx + nu)},
        mode="dense",
    )
    solver = setup_dense_solver(problem.n_dec, problem.n_ineq, problem.n_eq, problem.fixed_elements)
    return problem.with_solver(solver), nx, nu, N

# ======================================================================
# Helper: compute finite horizon LQR gain
# ======================================================================
def compute_finite_horizon_lqr_gain(
    A: jax.Array,
    B: jax.Array, 
    Q: jax.Array, 
    R: jax.Array, 
    N: int
) -> jax.Array:
    """Computes the analytical feedback gain K_0 for a finite-horizon LQR.
    
    This matches the MPC formulation where x_1 ... x_N are penalized by Q, 
    and u_0 ... u_{N-1} are penalized by R.
    """
    P = Q  # Terminal cost at x_N
    
    # Iterate backwards from k = N-1 down to 1 to find P_1 (the cost-to-go at x_1)
    for _ in range(N - 1):
        K = jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
        
    # Finally, compute K_0 which gives the optimal u_0 = -K_0 x_0
    K_0 = jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K_0

# ======================================================================
# TEST 1: Python For-Loop Equivalence and Cost Tracking
# ======================================================================
def test_closed_loop_lqr_equivalence():
    """Verify ClosedLoop matches a manual Python loop, computes finite horizon cost, 
    and outputs the exact analytical LQR optimal control law."""
    mpc, nx, nu, N = setup_lqr_mpc()
    N_sim = 15

    A_val = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B_val = jnp.array([[0.0], [0.1]])
    Q_val = jnp.eye(nx)
    R_val = jnp.eye(nu) * 0.1
    x0_val = jnp.array([-3.0, 1.0])

    plant = Dynamics(
        true_fun=lambda x, u, p: A_val @ x + B_val @ u,
        nominal_fun=lambda x, u, p: A_val @ x + B_val @ u,
        nx=nx,
        nu=nu
    )

    def init(inputs, n_steps):
        prep = mpc.prepare({"A": A_val, "B": B_val, "Q": Q_val, "R": R_val})
        return {"x": inputs["x0"], "cost_sum": 0.0}, {"prepared": prep}

    def step(state, k, constants):
        fast_v = {"x0": state["x"], "A": A_val}
        sol = mpc.solve_with_prepared(constants["prepared"], fast_v, None)
        
        u = sol["x"]["u0"]
        x_next = plant.step(state["x"], u, {}, {})
        
        step_cost = state["x"] @ Q_val @ state["x"] + u @ R_val @ u
        new_state = {"x": x_next, "cost_sum": state["cost_sum"] + step_cost}
        return new_state, {"x": state["x"], "u": u}

    sim = ClosedLoop(init=init, step=step, n_steps=N_sim, finalize=lambda c, l, cts: (c["cost_sum"],l))
    sim_out = sim.run({"x0": x0_val})

    # --- Analytical Benchmark ---
    K0 = compute_finite_horizon_lqr_gain(A_val, B_val, Q_val, R_val, N)
    
    prep_manual = mpc.prepare({"A": A_val, "B": B_val, "Q": Q_val, "R": R_val})
    x_curr = x0_val
    loop_xs, loop_us = [], []
    loop_total_cost = 0.0

    for _ in range(N_sim):
        sol = mpc.solve_with_prepared(prep_manual, {"x0": x_curr, "A": A_val}, None)
        u_curr = sol["x"]["u0"]
        
        # Verify the solver exactly matches the Riccati optimal control law
        np.testing.assert_allclose(u_curr, -K0 @ x_curr, atol=1e-5)
        
        loop_xs.append(x_curr)
        loop_us.append(u_curr)
        loop_total_cost += x_curr @ Q_val @ x_curr + u_curr @ R_val @ u_curr
        
        x_curr = A_val @ x_curr + B_val @ u_curr

    np.testing.assert_allclose(sim_out.trajectory["x"], jnp.stack(loop_xs), atol=1e-7)
    np.testing.assert_allclose(sim_out.trajectory["u"], jnp.stack(loop_us), atol=1e-7)
    np.testing.assert_allclose(float(sim_out.objective), loop_total_cost, atol=1e-7)


# ======================================================================
# TEST 2 & 3: Gradient correctness (Perfect Model & Model Mismatch)
# ======================================================================
@pytest.mark.parametrize("mismatch", [False, True])
def test_straight_through_gradients(mismatch):
    """Verify that gradients ALWAYS flow through the nominal model and match 
    the analytical Riccati closed-loop Jacobian."""
    mpc, nx, nu, N = setup_lqr_mpc()
    N_sim = 5

    A_nom = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B_nom = jnp.array([[0.0], [0.1]])
    Q_val = jnp.eye(nx)
    R_val = jnp.eye(nu) * 0.1
    
    A_true = jnp.array([[1.0, 0.2], [0.0, 0.8]]) if mismatch else A_nom

    plant = Dynamics(
        true_fun=lambda x, u, p: p["A"] @ x + B_nom @ u,
        nominal_fun=lambda x, u, p: A_nom @ x + B_nom @ u,
        nx=nx,
        nu=nu
    )

    def run_sim(x0):
        def init(inputs, n_steps):
            prep = mpc.prepare({"A": A_nom, "B": B_nom, "Q": Q_val, "R": R_val})
            return {"x": inputs["x0"]}, {"prepared": prep}

        def step(state, k, constants):
            fast_v = {"x0": state["x"], "A": A_nom}
            sol = mpc.solve_with_prepared(constants["prepared"], fast_v, None)
            x_next = plant.step(state["x"], sol["x"]["u0"], {"A": A_true}, {})
            return {"x": x_next}, None
            
        sim = ClosedLoop(init=init, step=step, n_steps=N_sim, finalize=lambda c, l, cts: (c["x"],l))
        return sim.run({"x0": x0}).objective

    # Use our custom vmap(jvp) helper instead of jacfwd
    jax_jacobian = _compute_jacobian(run_sim, jnp.array([1.0, 1.0]))

    # --- Compute theoretical expected Jacobian analytically via Riccati ---
    K0 = compute_finite_horizon_lqr_gain(A_nom, B_nom, Q_val, R_val, N)
    
    # The closed-loop state transition matrix is (A - B * K0)
    A_cl_nom = A_nom - B_nom @ K0
    
    # The analytical derivative of x_T with respect to x_0 is simply A_cl_nom^N
    expected_jacobian = jnp.linalg.matrix_power(A_cl_nom, N_sim)

    np.testing.assert_allclose(jax_jacobian, expected_jacobian, atol=1e-8)


# ======================================================================
# TEST 4: Stochastic True Dynamics (Noisy Model)
# ======================================================================
def test_noisy_true_model():
    """Verify the straight-through estimator survives stochastic noise injected into the true model
    and that its Jacobian exactly matches the analytical finite-horizon LQR."""
    mpc, nx, nu, N = setup_lqr_mpc()
    N_sim = 5
    
    A_nom = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B_nom = jnp.array([[0.0], [0.1]])
    Q_val = jnp.eye(nx)
    R_val = jnp.eye(nu)
    
    plant = Dynamics(
        true_fun=lambda x, u, p: A_nom @ x + B_nom @ u + p["noise"],
        nominal_fun=lambda x, u, p: A_nom @ x + B_nom @ u,
        nx=nx,
        nu=nu
    )

    def run_sim(x0):
        key = jax.random.PRNGKey(42)
        noise_seq = jax.random.normal(key, (N_sim, nx)) * 0.1

        def init(inputs, n_steps):
            prep = mpc.prepare({"A": A_nom, "B": B_nom, "Q": Q_val, "R": R_val})
            return {"x": inputs["x0"]}, {"prepared": prep, "noise": inputs["noise"]}

        def step(state, k, constants):
            fast_v = {"x0": state["x"], "A": A_nom}
            sol = mpc.solve_with_prepared(constants["prepared"], fast_v, None)
            
            current_noise = constants["noise"][k]
            x_next = plant.step(state["x"], sol["x"]["u0"], {"noise": current_noise}, {})
            return {"x": x_next}, None

        sim = ClosedLoop(init=init, step=step, n_steps=N_sim, finalize=lambda c, l, cts: (c["x"],None))
        return sim.run({"x0": x0, "noise": noise_seq}).objective

    # 1. Compute the Jacobian of the noisy simulation using AD
    # (Relies on the _compute_jacobian vmap(jvp) helper defined previously)
    jac_ad = _compute_jacobian(run_sim, jnp.array([1.0, 1.0]))
    
    # Ensure it didn't blow up / return NaNs
    assert not jnp.any(jnp.isnan(jac_ad))

    # 2. Compute the theoretical expected Jacobian analytically via Riccati
    K0 = compute_finite_horizon_lqr_gain(A_nom, B_nom, Q_val, R_val, N)
    
    # The closed-loop state transition matrix is (A - B * K0)
    A_cl_nom = A_nom - B_nom @ K0
    
    # The analytical derivative of x_T with respect to x_0 is simply A_cl_nom^T
    expected_jacobian = jnp.linalg.matrix_power(A_cl_nom, N_sim)

    # 3. Verify that the MPC solved the exact LQR problem, and the straight-through 
    # estimator correctly bypassed the stochastic noise to provide the exact gradient.
    np.testing.assert_allclose(jac_ad, expected_jacobian, atol=1e-8)