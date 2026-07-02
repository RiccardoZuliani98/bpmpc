"""Tests for the bpmpc MPC utility.

Covers a variety of MPC problem formulations by actively solving the 
resulting QPs using `jaxsparrow`:

1. **States + inputs as decision variables** (standard MPC layout)
   - Double integrator with and without box constraints
2. **Input-only decision variables** (condensed / shooting formulation)
   - State is eliminated; only u is optimized over
   - With and without box constraints
3. **Non-box constraints**
   - Polytopic state constraints (half-space intersections)
   - Mixed state-input coupling constraints (e.g. |u| <= f(x))
   - Terminal set constraint (ellipsoidal)
4. **Parametric cost and constraints**
   - Tracking a parametric reference trajectory
   - Time-varying bounds as slow parameters
5. **Slack / soft constraints**
   - Soft state bounds with quadratic + linear penalties
   - Partial slack (some rows hard, some soft)
6. **Edge cases / validation**
   - Mismatched n_var between cost and constraint
   - Undeclared variables
   - Equality + inequality mix
"""

from __future__ import annotations

from typing import Optional
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import scipy.sparse as sp

jax.config.update("jax_enable_x64", True)

from bpmpc.variable import Variable
from bpmpc.mpc import Cost, Constraint, SlackSpec, MPCProblem

# Use the real dense solver
from jaxsparrow import setup_dense_solver, setup_sparse_solver

# ======================================================================
# Helpers for Solver Attachment
# ======================================================================

def to_csc(jax_bcoo):
    """Converts a JAX BCOO array to a SciPy CSC matrix."""
    data = np.asarray(jax_bcoo.data)
    rows = np.asarray(jax_bcoo.indices[:, 0])
    cols = np.asarray(jax_bcoo.indices[:, 1])
    return sp.coo_matrix((data, (rows, cols)), shape=jax_bcoo.shape).tocsc()

def attach_solver(problem: MPCProblem, mode: str, dummy_slow_vars: Optional[dict] = None):
    """Attaches the correct dense or sparse solver based on the mode."""
    if dummy_slow_vars is None:
        dummy_slow_vars = {}
        
    if mode == "dense":
        solver = setup_dense_solver(
            n_var=problem.n_dec,
            n_ineq=problem.n_ineq,
            n_eq=problem.n_eq,
            fixed_elements=problem.fixed_elements,
        )
    else:
        # Extract the static BCOO arrays to get the sparsity pattern
        dummy_qp = problem.prepare(dummy_slow_vars)
        
        sparsity_patterns = {}
        if "P" not in problem.fixed_elements:
            sparsity_patterns["P"] = to_csc(dummy_qp.P)
        if "A" not in problem.fixed_elements:
            sparsity_patterns["A"] = to_csc(dummy_qp.A)
        if "G" not in problem.fixed_elements:
            sparsity_patterns["G"] = to_csc(dummy_qp.G)
            
        solver = setup_sparse_solver(
            n_var=problem.n_dec,
            n_ineq=problem.n_ineq,
            n_eq=problem.n_eq,
            fixed_elements=problem.fixed_elements,
            sparsity_patterns=sparsity_patterns,
        )
        
    return problem.with_solver(solver)


# ======================================================================
# Helpers
# ======================================================================

def todense(x):
    """Coerce BCOO → dense; no-op otherwise."""
    return x.todense() if isinstance(x, BCOO) else x

def _free_response(A_d, x0, N, nx):
    """Compute the free (unforced) response of the system."""
    x_free = jnp.zeros((N * nx,))
    curr_x = x0
    for k in range(N):
        curr_x = A_d @ curr_x
        x_free = x_free.at[k * nx:(k + 1) * nx].set(curr_x)
    return x_free


# ======================================================================
# 1. States + inputs as decision variables (full MPC)
# ======================================================================

@pytest.mark.parametrize("mode", ["dense", "sparse"])
class TestFullMPC:
    """Standard MPC: z = [x_1, ..., x_N, u_0, ..., u_{N-1}]."""

    nx, nu, N = 2, 1, 5

    def _build_double_integrator(self):
        """Build a minimal double-integrator MPC without helpers."""
        nx, nu, N = self.nx, self.nu, self.N
        n_var = N * nx + N * nu  # states x_1..x_N + inputs u_0..u_{N-1}

        A_d = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        B_d = jnp.array([[0.0], [1.0]])

        v_x0 = Variable("x0", (nx,))

        # -- Cost: sum_k ||x_k||^2_Q + ||u_k||^2_R --
        Q = jnp.eye(nx)
        R = 0.1 * jnp.eye(nu)

        Q_full = jnp.kron(jnp.eye(N), Q)
        R_full = jnp.kron(jnp.eye(N), R)
        P_mat = jnp.block([
            [Q_full, jnp.zeros((N * nx, N * nu))],
            [jnp.zeros((N * nu, N * nx)), R_full],
        ])

        cost = Cost(
            q_mat=lambda v: P_mat,
            v_in_q_mat=None,
        )

        n_eq = N * nx

        def dynamics_lhs(v):
            lhs = jnp.zeros((n_eq, n_var))
            for k in range(N):
                row_start = k * nx
                if k > 0:
                    col_x_prev = (k - 1) * nx
                    lhs = lhs.at[row_start:row_start + nx,
                                 col_x_prev:col_x_prev + nx].set(-A_d)
                col_x_next = k * nx
                lhs = lhs.at[row_start:row_start + nx,
                             col_x_next:col_x_next + nx].set(jnp.eye(nx))
                col_u = N * nx + k * nu
                lhs = lhs.at[row_start:row_start + nx,
                             col_u:col_u + nu].set(-B_d)
            return lhs

        def dynamics_rhs(v):
            x0 = v["x0"]
            rhs = jnp.zeros(n_eq)
            rhs = rhs.at[:nx].set(A_d @ x0)
            return rhs

        dynamics = Constraint(
            cst_type="equality",
            lhs=dynamics_lhs,
            rhs=dynamics_rhs,
            v_in_rhs={"x0": v_x0},
        )

        return cost, dynamics, v_x0, n_var

    def test_build_and_dimensions(self, mode):
        """Verify problem dimensions are correct."""
        cost, dynamics, v_x0, n_var = self._build_double_integrator()

        problem = MPCProblem(
            costs=[cost],
            constraints=[dynamics],
            mode=mode,
        )

        assert problem.n_var == n_var
        assert problem.n_eq == self.N * self.nx
        assert problem.n_ineq == 0
        assert problem.n_dec == n_var  # no slacks

    def test_equality_only_solve(self, mode):
        """Solve equality-constrained QP and verify dynamics are satisfied."""
        cost, dynamics, _, _ = self._build_double_integrator()

        problem = MPCProblem(
            costs=[cost],
            constraints=[dynamics],
            mode=mode,
        )
        
        mpc = attach_solver(problem, mode)

        x0 = jnp.array([1.0, 0.5])
        sol = mpc.solve({"x0": x0}, warmstart=None)
        z = sol["x"]

        # Check dynamics feasibility: A z = b
        qp = problem.solve_from(problem.prepare({}), {"x0": x0})
        
        A_dense = todense(qp.A)
        residual = A_dense @ z - todense(qp.b)
        np.testing.assert_allclose(np.array(residual), 0.0, atol=1e-8)

    def test_with_box_bounds(self, mode):
        """Add box constraints, solve, and verify bounds are respected."""
        nx, nu, N = self.nx, self.nu, self.N
        cost, dynamics, _, n_var = self._build_double_integrator()

        x_max, x_min = 5.0, -5.0
        u_max, u_min = 0.5, -0.5

        def bounds_lhs(v):
            I_x = jnp.eye(N * nx)
            I_u = jnp.eye(N * nu)
            top = jnp.block([[I_x, jnp.zeros((N * nx, N * nu))]])
            bot_x = jnp.block([[-I_x, jnp.zeros((N * nx, N * nu))]])
            top_u = jnp.block([[jnp.zeros((N * nu, N * nx)), I_u]])
            bot_u = jnp.block([[jnp.zeros((N * nu, N * nx)), -I_u]])
            return jnp.concatenate([top, bot_x, top_u, bot_u], axis=0)

        def bounds_rhs(v):
            ub_x = x_max * jnp.ones(N * nx)
            lb_x = -x_min * jnp.ones(N * nx)
            ub_u = u_max * jnp.ones(N * nu)
            lb_u = -u_min * jnp.ones(N * nu)
            return jnp.concatenate([ub_x, lb_x, ub_u, lb_u])

        bounds = Constraint("inequality", lhs=bounds_lhs, rhs=bounds_rhs)

        problem = MPCProblem(
            costs=[cost],
            constraints=[dynamics, bounds],
            mode=mode,
        )

        mpc = attach_solver(problem, mode)

        x0 = jnp.array([1.0, 0.5])
        sol = mpc.solve({"x0": x0}, warmstart=None)
        z = sol["x"]

        # Verify inequality feasibility: G z <= h
        qp = problem.solve_from(problem.prepare({}), {"x0": x0})
        
        G_dense = todense(qp.G)
        ineq_residual = G_dense @ z - todense(qp.h)
        assert np.all(np.array(ineq_residual) <= 1e-6), "Bounds violated!"
        
        # Verify equality constraints are still satisfied
        A_dense = todense(qp.A)
        eq_residual = A_dense @ z - todense(qp.b)
        np.testing.assert_allclose(np.array(eq_residual), 0.0, atol=1e-8)


# ======================================================================
# 2. Input-only decision variables (shooting / condensed MPC)
# ======================================================================

@pytest.mark.parametrize("mode", ["dense", "sparse"])
class TestInputOnlyMPC:
    """Condensed MPC: z = [u_0, ..., u_{N-1}], states are implicit."""

    nx, nu, N = 2, 1, 4

    def test_input_only_quadratic(self, mode):
        """Input-only QP with parametric linear cost from current state."""
        nx, nu, N = self.nx, self.nu, self.N
        n_var = N * nu

        A_d = jnp.array([[1.0, 0.1], [0.0, 1.0]])
        B_d = jnp.array([[0.0], [0.1]])
        Q = jnp.eye(nx)
        R = 0.01 * jnp.eye(nu)

        n_x_total = N * nx
        S = jnp.zeros((n_x_total, n_var))
        A_pow = [jnp.eye(nx)]
        for _ in range(N):
            A_pow.append(A_pow[-1] @ A_d)

        for k in range(N):
            for j in range(k + 1):
                row_s = k * nx
                col_s = j * nu
                S = S.at[row_s:row_s + nx, col_s:col_s + nu].set(
                    A_pow[k - j] @ B_d
                )

        Q_bar = jnp.kron(jnp.eye(N), Q)
        R_bar = jnp.kron(jnp.eye(N), R)

        H = S.T @ Q_bar @ S + R_bar

        v_x0 = Variable("x0", (nx,))

        cost = Cost(
            q_mat=lambda v: H,
            q_vec=lambda v: S.T @ Q_bar @ _free_response(A_d, v["x0"], N, nx),
            v_in_q_vec={"x0": v_x0},
        )

        u_max_val = 1.0

        bounds = Constraint(
            cst_type="inequality",
            lhs=lambda v: jnp.concatenate([jnp.eye(n_var), -jnp.eye(n_var)], axis=0),
            rhs=lambda v: u_max_val * jnp.ones(2 * n_var),
        )

        problem = MPCProblem(
            costs=[cost],
            constraints=[bounds],
            mode=mode,
        )

        mpc = attach_solver(problem, mode)

        x0 = jnp.array([2.0, -1.0])
        sol = mpc.solve({"x0": x0}, warmstart=None)
        z = sol["x"]

        # Check bounds: G z <= h
        qp = problem.solve_from(problem.prepare({}), {"x0": x0})
        
        # Safely convert to dense for numpy matrix multiplication
        G_dense = todense(qp.G)
        ineq_residual = G_dense @ z - todense(qp.h)
        assert np.all(np.array(ineq_residual) <= 1e-6)

    def test_input_only_no_constraints(self, mode):
        """Unconstrained input-only problem — verify analytic solve."""
        nx, nu, N = 2, 1, 3
        n_var = N * nu
        H = 2.0 * jnp.eye(n_var)
        q_vec_const = jnp.array([1.0, -1.0, 0.5])
        
        cost = Cost(
            q_mat=lambda v: H,
            q_vec=lambda v: q_vec_const,
        )

        problem = MPCProblem(
            costs=[cost],
            constraints=[],
            mode=mode,
        )
        assert problem.n_var == n_var
        assert problem.n_eq == 0
        assert problem.n_ineq == 0

        mpc = attach_solver(problem, mode)

        sol = mpc.solve({}, warmstart=None)
        
        # Analytically the unconstrained minimum is -H^{-1} q = -0.5 * q
        np.testing.assert_allclose(np.array(sol["x"]), -0.5 * np.array(q_vec_const), atol=1e-8)


# ======================================================================
# 3. Non-box constraints
# ======================================================================

@pytest.mark.parametrize("mode", ["dense", "sparse"])
class TestNonBoxConstraints:
    """Polytopic, coupling, and terminal constraints."""

    nx, nu, N = 2, 1, 3

    def test_polytopic_state_constraints(self, mode):
        """Solve with half-space state constraints: F x_k <= g."""
        nx, nu, N = self.nx, self.nu, self.N
        n_var = N * nx + N * nu

        n_faces = 4
        F = jnp.array([
            [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0],
        ])
        g = 3.0 * jnp.ones(n_faces)

        def polytope_lhs(v):
            lhs = jnp.zeros((N * n_faces, n_var))
            for k in range(N):
                row = k * n_faces
                col = k * nx
                lhs = lhs.at[row:row+n_faces, col:col+nx].set(F)
            return lhs

        def polytope_rhs(v):
            return jnp.tile(g, N)

        polytope = Constraint("inequality", lhs=polytope_lhs, rhs=polytope_rhs)
        cost = Cost(q_mat=lambda v: jnp.eye(n_var))

        problem = MPCProblem(
            costs=[cost],
            constraints=[polytope],
            mode=mode,
        )
        
        mpc = attach_solver(problem, mode)
        
        sol = mpc.solve({}, warmstart=None)
        z = sol["x"]
        
        qp = problem.solve_from(problem.prepare({}), {})
        
        # Safely convert to dense for numpy matrix multiplication
        G_dense = todense(qp.G)
        ineq_residual = G_dense @ z - todense(qp.h)
        assert np.all(np.array(ineq_residual) <= 1e-6)

    def test_mixed_state_input_coupling(self, mode):
        """Coupling constraint: u_k <= 0.5 * x_k."""
        nx, nu, N = 1, 1, 3
        n_var = N * nx + N * nu
        
        # Rewrite as: u_k - 0.5 x_k <= 0
        def coupling_lhs(v):
            lhs = jnp.zeros((N, n_var))
            for k in range(N):
                lhs = lhs.at[k, k * nx].set(-0.5)
                lhs = lhs.at[k, N * nx + k * nu].set(1.0)
            return lhs
            
        coupling = Constraint("inequality", lhs=coupling_lhs, rhs=lambda v: jnp.zeros(N))
        cost = Cost(q_mat=lambda v: jnp.eye(n_var))
        
        problem = MPCProblem(costs=[cost], constraints=[coupling], mode=mode)
        assert problem.n_ineq == N
        
        mpc = attach_solver(problem, mode)
        sol = mpc.solve({}, warmstart=None)
        z = sol["x"]
        
        qp = problem.solve_from(problem.prepare({}), {})
        
        G_dense = todense(qp.G)
        ineq_residual = G_dense @ z - todense(qp.h)
        assert np.all(np.array(ineq_residual) <= 1e-6)

    def test_terminal_ellipsoidal_constraint(self, mode):
        """Terminal constraint: H_term @ x_N <= h_term."""
        nx, nu, N = self.nx, self.nu, self.N
        n_var = N * nx + N * nu

        n_term_faces = 8
        angles = jnp.linspace(0, 2 * jnp.pi, n_term_faces, endpoint=False)
        H_term = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
        h_term = jnp.ones(n_term_faces)

        def terminal_lhs(v):
            lhs = jnp.zeros((n_term_faces, n_var))
            col_start = (N - 1) * nx
            lhs = lhs.at[:, col_start:col_start + nx].set(H_term)
            return lhs

        terminal = Constraint("inequality", lhs=terminal_lhs, rhs=lambda v: h_term)
        cost = Cost(q_mat=lambda v: jnp.eye(n_var))

        problem = MPCProblem(costs=[cost], constraints=[terminal], mode=mode)
        assert problem.n_ineq == n_term_faces
        
        mpc = attach_solver(problem, mode)
        sol = mpc.solve({}, warmstart=None)
        z = sol["x"]
        
        qp = problem.solve_from(problem.prepare({}), {})
        
        G_dense = todense(qp.G)
        ineq_residual = G_dense @ z - todense(qp.h)
        assert np.all(np.array(ineq_residual) <= 1e-6)


# ======================================================================
# 4. Parametric cost and constraints
# ======================================================================

@pytest.mark.parametrize("mode", ["dense", "sparse"])
class TestParametricMPC:
    """Problems driven by reference tracking."""

    def test_parametric_cost_reference_tracking(self, mode):
        nx, nu, N = 2, 2, 5  # nu=2 so system is fully decoupled and controllable
        n_var = N * nx + N * nu

        v_xref = Variable("x_ref", (N, nx))
        v_x0 = Variable("x0", (nx,))

        Q_full = jnp.kron(jnp.eye(N), jnp.eye(nx))
        R_full = jnp.kron(jnp.eye(N), 0.1 * jnp.eye(nu))
        P_mat = jnp.block([
            [Q_full, jnp.zeros((N * nx, N * nu))],
            [jnp.zeros((N * nu, N * nx)), R_full],
        ])

        def tracking_q(v):
            x_ref_flat = v["x_ref"].flatten()
            q_x = -Q_full @ x_ref_flat
            q_u = jnp.zeros(N * nu)
            return jnp.concatenate([q_x, q_u])

        cost = Cost(
            q_mat=lambda v: P_mat,
            q_vec=tracking_q,
            v_in_q_vec={"x_ref": v_xref},
        )

        A_d = jnp.eye(nx)
        B_d = jnp.eye(nx)  # Fully controllable decoupled system
        n_eq = N * nx

        def dyn_lhs(v):
            lhs = jnp.zeros((n_eq, n_var))
            for k in range(N):
                rs = k * nx
                if k > 0:
                    lhs = lhs.at[rs:rs + nx, (k - 1) * nx:k * nx].set(-A_d)
                lhs = lhs.at[rs:rs + nx, k * nx:(k + 1) * nx].set(jnp.eye(nx))
                lhs = lhs.at[rs:rs + nx, N * nx + k * nu:N * nx + (k + 1) * nu].set(-B_d)
            return lhs

        def dyn_rhs(v):
            rhs = jnp.zeros(n_eq)
            rhs = rhs.at[:nx].set(A_d @ v["x0"])
            return rhs

        dynamics = Constraint("equality", dyn_lhs, dyn_rhs, v_in_rhs={"x0": v_x0})

        problem = MPCProblem(
            costs=[cost],
            constraints=[dynamics],
            mode=mode,
            outputs={"states": slice(0, N * nx)}
        )

        mpc = attach_solver(problem, mode)

        # Track a target coordinate of [5.0, 5.0]
        ref_a = 5.0 * jnp.ones((N, nx))
        x0 = jnp.zeros(nx)

        sol = mpc.solve({"x0": x0, "x_ref": ref_a}, warmstart=None)
        states = sol["x"]["states"].reshape(N, nx)
        
        np.testing.assert_allclose(np.array(states[-1]), np.array(ref_a[-1]), atol=1e-3)

    def test_time_varying_bounds_as_slow_parameters(self, mode):
        """Pass variable bounds efficiently as slow constraints."""
        nx, nu, N = 1, 1, 3
        n_var = N * nx + N * nu
        
        v_ub = Variable("ub", (N,))
        def bounds_lhs(v):
            return jnp.eye(n_var)
            
        def bounds_rhs(v):
            # Only bound the states. Leave inputs effectively unconstrained here.
            return jnp.concatenate([v["ub"], 100.0 * jnp.ones(N * nu)])
            
        bounds = Constraint("inequality", lhs=bounds_lhs, rhs=bounds_rhs, v_in_rhs={"ub": v_ub})
        
        # Minimize 0.5*x^2 - 10x -> unconstrained minimum is at x=10
        # This guarantees the state will push all the way to its upper bounds (1, 2, and 3)
        cost = Cost(
            q_mat=lambda v: jnp.eye(n_var), 
            q_vec=lambda v: -10.0 * jnp.ones(n_var)
        )
        
        problem = MPCProblem(costs=[cost], constraints=[bounds], slow_vars=["ub"], mode=mode)
        
        # Provide dummy data for the slow variable so the sparse builder can probe it
        dummy_ub = jnp.ones(N)
        mpc = attach_solver(problem, mode, dummy_slow_vars={"ub": dummy_ub})
        
        ub_val = jnp.array([1.0, 2.0, 3.0])
        prepared = mpc.prepare({"ub": ub_val})
        sol = mpc.solve_with_prepared(fast_v={}, prepared_qp=prepared, warmstart=None)
        
        states = sol["x"][:N * nx]
        np.testing.assert_allclose(np.array(states), np.array(ub_val), atol=1e-5)


# ======================================================================
# 5. Slack / soft constraints
# ======================================================================

@pytest.mark.parametrize("mode", ["dense", "sparse"])
class TestSlackConstraints:
    """Soft constraints with quadratic and linear penalties."""

    def test_full_slack_all_rows(self, mode):
        """Force a bound violation using conflicting hard and soft constraints."""
        n_var = 2
        n_ineq = 2
        n_slack = 2

        # Cost: minimize x_0^2 + x_1^2 (wants x = [0, 0])
        cost = Cost(q_mat=lambda v: 2.0 * jnp.eye(n_var))
        
        # Hard equality constraint: x_0 = 0, x_1 = 0
        eq = Constraint(
            "equality", 
            lhs=lambda v: jnp.eye(n_var), 
            rhs=lambda v: jnp.zeros(n_var)
        )

        # Soft inequality constraint: x_0 >= 5, x_1 >= 10 (formulated as -x <= -rhs)
        # Because x=[0,0] is hard, this constraint cannot be satisfied without slacks.
        slack = SlackSpec.slack_all(n_ineq, w_quad=10.0, w_lin=1.0)
        bounds = Constraint(
            cst_type="inequality",
            lhs=lambda v: -jnp.eye(n_ineq),
            rhs=lambda v: jnp.array([-5.0, -10.0]),
            slack=slack,
        )

        problem = MPCProblem(
            costs=[cost],
            constraints=[eq, bounds],
            outputs={"x": slice(0, 2), "s": slice(2, 4)},
            mode=mode,
        )

        # Attach the proper solver (sparse or dense) automatically
        mpc = attach_solver(problem, mode)

        sol = mpc.solve({}, warmstart=None)
        
        # x must be strictly [0.0, 0.0] to satisfy the hard equality constraint
        np.testing.assert_allclose(sol["x"]["x"], np.array([0.0, 0.0]), atol=1e-6)
        
        # The inequality requires -x - s <= [-5, -10]. 
        # Since x=0, this simplifies to -s <= [-5, -10], meaning s >= [5, 10].
        # Since we penalize the slack heavily, the solver will settle exactly at the boundaries.
        np.testing.assert_allclose(sol["x"]["s"], np.array([5.0, 10.0]), atol=1e-5)

    def test_partial_slack(self, mode):
        """Some rows hard, some rows soft, with conflicting bounds."""
        n_var = 2
        n_slack = 1

        # Cost: minimize x_0^2 + x_1^2 (wants x = [0, 0])
        cost = Cost(q_mat=lambda v: 2.0 * jnp.eye(n_var))
        
        # Inequality constraints:
        # Row 0 (Hard):  x_0 <= 1
        # Row 1 (Soft):  x_0 >= 5  --> -x_0 <= -5  (Conflicts with Row 0)
        # Row 2 (Hard):  x_1 >= 2  --> -x_1 <= -2  (No conflict, just a standard hard bound)
        slack = SlackSpec.slack_rows(3, rows=[1], w_quad=10.0, w_lin=1.0)
        
        def bounds_lhs(v):
            return jnp.array([
                [ 1.0,  0.0],  # Row 0
                [-1.0,  0.0],  # Row 1
                [ 0.0, -1.0]   # Row 2
            ])
            
        def bounds_rhs(v):
            return jnp.array([1.0, -5.0, -2.0])
            
        bounds = Constraint("inequality", lhs=bounds_lhs, rhs=bounds_rhs, slack=slack)
        
        problem = MPCProblem(
            costs=[cost],
            constraints=[bounds],
            outputs={"x": slice(0, 2), "s": slice(2, 3)},
            mode=mode
        )
        
        # Total decision variables = 2 base vars + 1 slack var = 3
        assert problem.n_dec == n_var + 1
        assert problem.n_ineq == 3 + n_slack
        
        # Attach the proper solver (sparse or dense) automatically
        mpc = attach_solver(problem, mode)
        sol = mpc.solve({}, warmstart=None)
        
        x = np.array(sol["x"]["x"])
        s = float(sol["x"]["s"][0])
        
        # Analysis of optimal solution:
        # x_1 wants to be 0, bounded by x_1 >= 2. It goes to 2.0.
        # x_0 wants to be 0, soft bounded by x_0 >= 5, hard bounded by x_0 <= 1.
        # To minimize the slack variable (s_0 >= 5 - x_0), x_0 will go as high as the hard bound allows.
        # Thus, x_0 = 1.0. This leaves a slack violation of 4.0 (i.e. s = 4.0).
        np.testing.assert_allclose(x, np.array([1.0, 2.0]), atol=1e-5)
        np.testing.assert_allclose(s, 4.0, atol=1e-5)


# ======================================================================
# 6. Edge cases / validation
# ======================================================================

@pytest.mark.parametrize("mode", ["dense", "sparse"])
class TestEdgeCases:

    def test_mismatched_n_var(self, mode):
        """Mismatching dimensions between costs and constraints."""
        cost = Cost(q_mat=lambda v: jnp.eye(3))
        cst = Constraint("equality", lhs=lambda v: jnp.ones((2, 4)), rhs=lambda v: jnp.zeros(2))
        with pytest.raises(ValueError, match="Inconsistent n_var"):
            MPCProblem(costs=[cost], constraints=[cst], mode=mode)

    def test_undeclared_variable_in_slow_vars_raises(self, mode):
        """Variables passed to slow_vars that aren't defined in sets should fail."""
        v = Variable("mystery", (3,))
        cost = Cost(
            q_mat=lambda v: jnp.eye(3),
            q_vec=lambda v: v["mystery"],
            v_in_q_vec={"mystery": v},
        )
        with pytest.raises(ValueError, match="unknown names"):
            MPCProblem(
                costs=[cost], constraints=[],
                slow_vars=["not_a_mystery"],
                mode=mode
            )

    def test_no_costs_raises(self, mode):
        """At least one cost is required."""
        with pytest.raises(ValueError, match="At least one cost is required"):
            MPCProblem(costs=[], constraints=[], mode=mode)

    def test_equality_and_inequality_mix(self, mode):
        """Problem with both equality and inequality constraints."""
        n_var = 4
        cost = Cost(q_mat=lambda v: jnp.eye(n_var))
        eq = Constraint("equality", lhs=lambda v: jnp.ones((2, n_var)), rhs=lambda v: jnp.zeros(2))
        ineq = Constraint("inequality", lhs=lambda v: jnp.eye(n_var), rhs=lambda v: jnp.ones(n_var))

        problem = MPCProblem(
            costs=[cost],
            constraints=[eq, ineq],
            mode=mode
        )

        assert problem.n_eq == 2
        assert problem.n_ineq == n_var

    def test_multiple_costs_sum(self, mode):
        """Multiple costs should be summed element-wise."""
        n_var = 2
        cost1 = Cost(q_mat=lambda v: jnp.eye(n_var))
        cost2 = Cost(q_mat=lambda v: 2.0 * jnp.eye(n_var))

        problem = MPCProblem(costs=[cost1, cost2], constraints=[], mode=mode)
        qp = problem.solve_from(problem.prepare({}), {})
        
        # Safely convert to dense if the backend returned a BCOO array
        P_dense = qp.P.todense() if hasattr(qp.P, "todense") else qp.P
        np.testing.assert_allclose(np.array(P_dense), 3.0 * np.eye(n_var))