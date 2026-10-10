"""Tests for MPC helper factories and builders.

Covers: lti_dynamics, ltv_dynamics, nonlinear_dynamics, box_bounds,
        state_tracking_cost, output_tracking_cost,
        and all corresponding build_* functions.
"""

from __future__ import annotations

import jax
import numpy as np
import jax.numpy as jnp
import pytest

from bpmpc.variable import Variable
from bpmpc.mpc.helpers import (
    lti_dynamics, ltv_dynamics, nonlinear_dynamics,
    box_bounds,
    state_tracking_cost, output_tracking_cost,
    build_lti_lhs, build_lti_rhs,
    build_ltv_lhs, build_ltv_rhs,
    build_box_lhs, build_box_rhs,
    build_state_tracking, build_output_tracking,
)
from bpmpc.mpc import SlackSpec


# ======================================================================
# Shared fixtures
# ======================================================================

NX, NU, N = 2, 1, 5
DT = 0.1

A_LTI = jnp.array([[1.0, DT], [0.0, 1.0]])
B_LTI = jnp.array([[0.5 * DT**2], [DT]])

# LTV: stack N copies (same as LTI for comparison)
A_LTV = jnp.broadcast_to(A_LTI, (N, NX, NX))
B_LTV = jnp.broadcast_to(B_LTI, (N, NX, NU))

X0_VAL = jnp.array([1.0, 0.0])

X_MIN = jnp.array([-5.0, -5.0])
X_MAX = jnp.array([ 5.0,  5.0])
U_MIN = jnp.array([-1.0])
U_MAX = jnp.array([ 1.0])

N_VAR = N * NX + N * NU
N_EQ  = N * NX
N_INEQ = 2 * (N * NX + N * NU)


# ======================================================================
# Dynamics
# ======================================================================

class TestLTIDynamics:

    def test_lti_builder(self):
        lhs = build_lti_lhs(A_LTI, B_LTI, N).todense()
        assert lhs.shape == (N_EQ, N_VAR)
        
        # Verify block structure
        I_nx = jnp.eye(NX)
        assert jnp.allclose(lhs[:NX, :NX], -I_nx)
        assert jnp.allclose(lhs[NX:2*NX, :NX], A_LTI)

        rhs = build_lti_rhs(A_LTI, X0_VAL, N)
        assert rhs.shape == (N_EQ,)
        assert jnp.allclose(rhs[:NX], -A_LTI @ X0_VAL)
        assert jnp.allclose(rhs[NX:], 0.0)

    def test_lti_factory(self):
        v_x0 = Variable("x0", (NX,))
        dyn = lti_dynamics(A_LTI, B_LTI, v_x0, N)

        assert dyn.is_equality
        assert not dyn.is_parametric_lhs
        assert dyn.is_parametric_rhs
        assert "x0" in dyn.v_in_rhs

        lhs_num = dyn.eval_lhs({})
        rhs_num = dyn.eval_rhs({"x0": X0_VAL})

        assert lhs_num.shape == (N_EQ, N_VAR)
        assert rhs_num.shape == (N_EQ,)

    def test_trajectory_satisfaction(self):
        """Verify that a valid trajectory perfectly satisfies lhs @ z == rhs."""
        U = jnp.ones((N, NU))
        X = [X0_VAL]
        for k in range(N):
            X.append(A_LTI @ X[-1] + B_LTI @ U[k])
        X = jnp.stack(X)
        
        # Pack trajectory into the decision vector z
        z = jnp.concatenate([X[1:].reshape(-1), U.reshape(-1)])

        dyn = lti_dynamics(A_LTI, B_LTI, X0_VAL, N) # X0 as constant for easy eval
        lhs = dyn.eval_lhs({})
        rhs = dyn.eval_rhs({})

        assert jnp.allclose(lhs @ z, rhs, atol=1e-6)


class TestLTVDynamics:

    def test_ltv_builder(self):
        lhs = build_ltv_lhs(A_LTV, B_LTV, N)
        assert lhs.shape == (N_EQ, N_VAR)
        
        rhs = build_ltv_rhs(A_LTV, X0_VAL, N)
        assert rhs.shape == (N_EQ,)
        assert jnp.allclose(rhs[:NX], -A_LTV[0] @ X0_VAL)

    def test_ltv_factory(self):
        v_x0 = Variable("x0", (NX,))
        dyn = ltv_dynamics(A_LTV, B_LTV, v_x0, N)

        assert dyn.is_equality
        assert not dyn.is_parametric_lhs
        
        lhs_num = dyn.eval_lhs({})
        rhs_num = dyn.eval_rhs({"x0": X0_VAL})
        
        assert lhs_num.shape == (N_EQ, N_VAR)
        assert rhs_num.shape == (N_EQ,)

    def test_ltv_matches_lti(self):
        """Verify auto-tiling a 2D matrix into LTV matches explicit LTI."""
        v_x0 = Variable("x0", (NX,))
        dyn_ltv = ltv_dynamics(A_LTI, B_LTI, v_x0, horizon=N)
        dyn_lti = lti_dynamics(A_LTI, B_LTI, v_x0, horizon=N)
        
        lhs_ltv = dyn_ltv.eval_lhs({})
        rhs_ltv = dyn_ltv.eval_rhs({"x0": X0_VAL})
        
        lhs_lti = dyn_lti.eval_lhs({})
        rhs_lti = dyn_lti.eval_rhs({"x0": X0_VAL})
        
        assert jnp.allclose(lhs_ltv, lhs_lti)
        assert jnp.allclose(rhs_ltv, rhs_lti)

    def test_trajectory_satisfaction(self):
        """Verify that a valid time-varying trajectory satisfies lhs @ z == rhs."""
        A_seq = jnp.array([A_LTI * (1 + 0.1 * k) for k in range(N)])
        B_seq = jnp.array([B_LTI * (1 - 0.05 * k) for k in range(N)])

        U = jnp.ones((N, NU))
        X = [X0_VAL]
        for k in range(N):
            X.append(A_seq[k] @ X[-1] + B_seq[k] @ U[k])
        X = jnp.stack(X)
        z = jnp.concatenate([X[1:].reshape(-1), U.reshape(-1)])

        dyn = ltv_dynamics(A_seq, B_seq, X0_VAL, N)
        lhs = dyn.eval_lhs({})
        rhs = dyn.eval_rhs({})

        assert jnp.allclose(lhs @ z, rhs, atol=1e-6)


class TestNonlinearDynamics:

    def setup_method(self):
        from bpmpc.dynamics import Dynamics
        
        def nl_step(x, u, p):
            # A distinct nonlinear dynamics function
            return A_LTI @ x + B_LTI @ u + 0.1 * jnp.sin(x)
            
        self.nl_step = nl_step
        self.dyn = Dynamics(nominal_fun=nl_step, true_fun=nl_step, nx=NX, nu=NU)

    def test_nonlinear_dynamics_factory(self):
        v_x_nom = Variable("x_nom", (N, NX))
        v_u_nom = Variable("u_nom", (N, NU))

        cst = nonlinear_dynamics(self.dyn, v_x_nom, v_u_nom, horizon=N)

        assert cst.is_equality
        assert cst.is_parametric
        assert "x_nom" in cst.v_in_lhs
        assert "u_nom" in cst.v_in_lhs

        x_val = jnp.zeros((N, NX))
        u_val = jnp.zeros((N, NU))

        lhs_num = cst.eval_lhs({"x_nom": x_val, "u_nom": u_val})
        rhs_num = cst.eval_rhs({"x_nom": x_val, "u_nom": u_val})

        assert lhs_num.shape == (N * NX, N * NX + N * NU)
        assert rhs_num.shape == (N * NX,)

    def test_point_matches_lti(self):
        """Verify linearizing around a single point matches manual LTI formulation."""
        x_eq = jnp.array([0.5, -0.2])
        u_eq = jnp.array([1.0])
        
        # Linearize manually
        A_eq = jax.jacfwd(self.nl_step, 0)(x_eq, u_eq, {})
        B_eq = jax.jacfwd(self.nl_step, 1)(x_eq, u_eq, {})
        c_eq = self.nl_step(x_eq, u_eq, {}) - A_eq @ x_eq - B_eq @ u_eq

        # LTI expects a separate x0. Nonlinear uses x_eq as the constant trajectory.
        cst_lti = lti_dynamics(A_eq, B_eq, x_eq, horizon=N, c=c_eq)
        cst_nl = nonlinear_dynamics(self.dyn, x_eq, u_eq, horizon=N)

        lhs_lti = cst_lti.eval_lhs({})
        rhs_lti = cst_lti.eval_rhs({})
        
        lhs_nl = cst_nl.eval_lhs({})
        rhs_nl = cst_nl.eval_rhs({})

        assert jnp.allclose(lhs_nl, lhs_lti, atol=1e-5)
        assert jnp.allclose(rhs_nl, rhs_lti, atol=1e-5)

    def test_trajectory_matches_ltv(self):
        """Verify linearizing along a trajectory matches manual LTV formulation."""
        key = jax.random.PRNGKey(42)
        X_traj = jax.random.normal(key, (N, NX))
        U_traj = jax.random.normal(key, (N, NU))

        # Linearize along traj
        def lin_step(x, u):
            A = jax.jacfwd(self.nl_step, 0)(x, u, {})
            B = jax.jacfwd(self.nl_step, 1)(x, u, {})
            c = self.nl_step(x, u, {}) - A @ x - B @ u
            return A, B, c
        
        A_traj, B_traj, c_traj = jax.vmap(lin_step)(X_traj, U_traj)

        # LTV x0 is X_traj[0]
        cst_ltv = ltv_dynamics(A_traj, B_traj, X_traj[0], horizon=N, c=c_traj)
        cst_nl = nonlinear_dynamics(self.dyn, X_traj, U_traj, horizon=N)

        lhs_ltv = cst_ltv.eval_lhs({})
        rhs_ltv = cst_ltv.eval_rhs({})

        lhs_nl = cst_nl.eval_lhs({})
        rhs_nl = cst_nl.eval_rhs({})

        assert jnp.allclose(lhs_nl, lhs_ltv, atol=1e-5)
        assert jnp.allclose(rhs_nl, rhs_ltv, atol=1e-5)

    def test_trajectory_satisfaction(self):
        """Verify that simulating the nonlinear system exactly satisfies its linearized constraints."""
        U = jnp.ones((N, NU))
        X = [X0_VAL]
        for k in range(N):
            X.append(self.nl_step(X[-1], U[k], {}))
        X = jnp.stack(X)
        
        # We linearize *along this exact generated trajectory*
        X_nom = X[:-1]
        U_nom = U
        
        z = jnp.concatenate([X[1:].reshape(-1), U.reshape(-1)])

        cst = nonlinear_dynamics(self.dyn, X_nom, U_nom, horizon=N)
        lhs = cst.eval_lhs({})
        rhs = cst.eval_rhs({})

        assert jnp.allclose(lhs @ z, rhs, atol=1e-5)


# ======================================================================
# Bounds
# ======================================================================

class TestBoxBounds:

    def test_box_bounds_builder(self):
        lhs = build_box_lhs(NX, NU, N)
        assert lhs.shape == (N_INEQ, N_VAR)

        rhs = build_box_rhs(X_MIN, X_MAX, U_MIN, U_MAX, N)
        assert rhs.shape == (N_INEQ,)

        nx_tot = N * NX
        assert jnp.allclose(rhs[:nx_tot], jnp.tile(X_MAX, N))
        assert jnp.allclose(rhs[nx_tot:2*nx_tot], jnp.tile(-X_MIN, N))

    def test_box_bounds_factory(self):
        v_xmax = Variable("xmax", (NX,))
        bounds = box_bounds(horizon=N, x_min=X_MIN, x_max=v_xmax, u_min=U_MIN, u_max=U_MAX)

        assert not bounds.is_equality
        assert not bounds.is_parametric_lhs
        assert bounds.is_parametric_rhs
        assert "xmax" in bounds.v_in_rhs

        lhs_num = bounds.eval_lhs({})
        rhs_num = bounds.eval_rhs({"xmax": X_MAX})

        assert lhs_num.shape == (N_INEQ, N_VAR)
        assert rhs_num.shape == (N_INEQ,)

    def test_box_bounds_dropped_rows_match_masked_full(self):
        """Row-selected LHS must equal masking the full [Fx; -Fx; Fu; -Fu]."""
        x_max = jnp.array([5.0, jnp.inf])       # second state upper bound dropped
        u_min = jnp.array([-jnp.inf])           # input lower bound dropped
        bounds = box_bounds(horizon=N, x_min=X_MIN, x_max=x_max,
                            u_min=u_min, u_max=U_MAX)

        keep = jnp.concatenate([
            jnp.tile(jnp.isfinite(x_max), N),
            jnp.tile(jnp.isfinite(X_MIN), N),
            jnp.tile(jnp.isfinite(U_MAX), N),
            jnp.tile(jnp.isfinite(u_min), N),
        ])
        expected_lhs = build_box_lhs(NX, NU, N).todense()[keep]
        expected_rhs = build_box_rhs(X_MIN, x_max, u_min, U_MAX, N)[keep]

        assert jnp.array_equal(bounds.eval_lhs({}), expected_lhs)
        assert jnp.array_equal(bounds.eval_rhs({}), expected_rhs)
        assert bounds.n_cst == int(keep.sum())

    def test_box_bounds_partial(self):
        # Only constrain U, leave X unbounded
        bounds = box_bounds(horizon=N, u_min=U_MIN, u_max=U_MAX, n_x=NX)
        
        rhs_num = bounds.eval_rhs({})
        lhs_num = bounds.eval_lhs({})
        # n_x_total = N * NX
        
        # x_max_vec = rhs_num[:n_x_total]
        # minus_x_min_vec = rhs_num[n_x_total : 2*n_x_total]
        
        # assert jnp.all(jnp.isinf(x_max_vec))
        # assert jnp.all(jnp.isinf(minus_x_min_vec))

        total_csts = NU*2*N
        total_var = (NX+NU)*N
        assert rhs_num.shape == (total_csts,)
        assert lhs_num.shape == (total_csts,total_var)

        expected_elements = jnp.concatenate([U_MAX, -U_MIN]) 
        actual_unique = jnp.unique(rhs_num)
        expected_unique = jnp.unique(expected_elements)
        np.testing.assert_allclose(actual_unique, expected_unique)


    def test_box_bounds_satisfies_interior(self):
        """Verify that a sequence strictly inside the bounds satisfies lhs @ z < rhs."""
        bounds = box_bounds(horizon=N, x_min=X_MIN, x_max=X_MAX, u_min=U_MIN, u_max=U_MAX)
        lhs = bounds.eval_lhs({})
        rhs = bounds.eval_rhs({})

        # Origin is strictly between X_MIN/MAX [-5, 5] and U_MIN/MAX [-1, 1]
        X_seq = jnp.zeros((N, NX))
        U_seq = jnp.zeros((N, NU))
        
        # Pack into decision vector z
        z = jnp.concatenate([X_seq.reshape(-1), U_seq.reshape(-1)])

        val = lhs @ z
        # Condition is val <= rhs. It should be strictly satisfied.
        assert jnp.all(val <= rhs)
        assert jnp.all(val < rhs)

    def test_box_bounds_violates_exterior(self):
        """Verify that a sequence outside the bounds violates lhs @ z <= rhs."""
        bounds = box_bounds(horizon=N, x_min=X_MIN, x_max=X_MAX, u_min=U_MIN, u_max=U_MAX)
        lhs = bounds.eval_lhs({})
        rhs = bounds.eval_rhs({})

        # Exceed X_MAX by 1.0, and fall below U_MIN by 1.0
        X_seq = jnp.tile(X_MAX + 1.0, (N, 1))
        U_seq = jnp.tile(U_MIN - 1.0, (N, 1))
        
        z = jnp.concatenate([X_seq.reshape(-1), U_seq.reshape(-1)])

        val = lhs @ z
        
        # It must violate the constraint, meaning some elements are > rhs
        assert jnp.any(val > rhs)

    def test_box_bounds_satisfies_boundary(self):
        """Verify that a sequence exactly on the min/max bounds is perfectly tight."""
        bounds = box_bounds(horizon=N, x_min=X_MIN, x_max=X_MAX, u_min=U_MIN, u_max=U_MAX)
        lhs = bounds.eval_lhs({})
        rhs = bounds.eval_rhs({})

        # Alternate hitting the MAX limit and MIN limit at each timestep
        X_seq = jnp.array([X_MAX if i % 2 == 0 else X_MIN for i in range(N)])
        U_seq = jnp.array([U_MAX if i % 2 == 0 else U_MIN for i in range(N)])
        
        z = jnp.concatenate([X_seq.reshape(-1), U_seq.reshape(-1)])

        val = lhs @ z

        # Condition <= rhs must hold everywhere (accounting for minor float precision)
        assert jnp.all(val <= rhs + 1e-6)
        
        # Since we are riding the boundaries, exactly half the inequalities 
        # (the active ones) should be perfectly tight (val == rhs)
        assert jnp.any(jnp.isclose(val, rhs))


# ======================================================================
# Costs
# ======================================================================

class TestCosts:

    def test_state_tracking_builder(self):
        Q_seq = jnp.broadcast_to(jnp.eye(NX), (N + 1, NX, NX))
        R_seq = jnp.broadcast_to(0.1 * jnp.eye(NU), (N, NU, NU))
        rx = jnp.zeros((N + 1, NX))
        ru = jnp.zeros((N, NU))

        P, q, c = build_state_tracking(Q_seq, R_seq, rx, ru, X0_VAL, N)
        P = P.todense()

        assert P.shape == (N_VAR, N_VAR)
        assert q.shape == (N_VAR,)
        assert c.shape == ()

        assert jnp.allclose(P, P.T)
        evals = jnp.linalg.eigvalsh(P)
        assert jnp.all(evals >= -1e-10)

    def test_state_tracking_factory(self):
        Q_seq = jnp.broadcast_to(jnp.eye(NX), (N + 1, NX, NX))
        R_seq = jnp.broadcast_to(0.1 * jnp.eye(NU), (N, NU, NU))
        rx = jnp.zeros((N + 1, NX))
        ru = jnp.zeros((N, NU))
        
        v_x0 = Variable("x0", (NX,))
        cost = state_tracking_cost(Q_seq, R_seq, rx, ru, v_x0, N)

        assert cost.is_parametric
        # assert "x0" in cost.v_in_q_vec
        assert "x0" in cost.v_in_c

        q_mat = cost.eval_q_mat({})
        q_vec = cost.eval_q_vec({"x0": X0_VAL})
        c_val = cost.eval_c({"x0": X0_VAL})

        assert q_mat.shape == (N_VAR, N_VAR)
        assert q_vec.shape == (N_VAR,)
        assert c_val.shape == ()

    def test_state_tracking_auto_tiling(self):
        v_x0 = Variable("x0", (NX,))
        Q_mat = jnp.eye(NX)
        R_mat = jnp.eye(NU)
        r_x_vec = jnp.zeros(NX)
        r_u_vec = jnp.zeros(NU)
        
        cost = state_tracking_cost(Q_mat, R_mat, r_x_vec, r_u_vec, v_x0, horizon=N)
        q_mat_eval = cost.eval_q_mat({"x0": X0_VAL})
        
        assert q_mat_eval.shape == (N_VAR, N_VAR)

    def test_state_tracking_auto_tiling_equivalence(self):
        """Verify passing 2D matrices / 1D vectors yields the exact same cost as passing broadcasted sequences."""
        v_x0 = Variable("x0", (NX,))
        Q_mat = jnp.eye(NX) * 2.0
        R_mat = jnp.eye(NU) * 0.5
        r_x_vec = jnp.ones(NX) * 0.1
        r_u_vec = jnp.ones(NU) * 0.2

        # 1. Provide single constant elements (relies on auto_tile)
        cost_auto = state_tracking_cost(Q_mat, R_mat, r_x_vec, r_u_vec, v_x0, N)
        
        # 2. Provide explicitly tiled sequences
        Q_seq = jnp.broadcast_to(Q_mat, (N + 1, NX, NX))
        R_seq = jnp.broadcast_to(R_mat, (N, NU, NU))
        rx_seq = jnp.broadcast_to(r_x_vec, (N + 1, NX))
        ru_seq = jnp.broadcast_to(r_u_vec, (N, NU))
        cost_seq = state_tracking_cost(Q_seq, R_seq, rx_seq, ru_seq, v_x0, N)

        vars_dict = {"x0": X0_VAL}
        
        assert jnp.allclose(cost_auto.eval_q_mat(vars_dict), cost_seq.eval_q_mat(vars_dict))
        assert jnp.allclose(cost_auto.eval_q_vec(vars_dict), cost_seq.eval_q_vec(vars_dict))
        assert jnp.allclose(cost_auto.eval_c(vars_dict), cost_seq.eval_c(vars_dict))

    def test_state_tracking_loop_equivalence(self):
        """Verify the block matrix formulation matches a standard for-loop accumulation."""
        # Setup time-varying sequences
        Q_seq = jnp.array([jnp.eye(NX) * (i + 1) for i in range(N + 1)])
        R_seq = jnp.array([jnp.eye(NU) * (i + 1) for i in range(N)])
        rx_seq = jnp.array([jnp.ones(NX) * i for i in range(N + 1)])
        ru_seq = jnp.array([jnp.ones(NU) * i for i in range(N)])

        cost = state_tracking_cost(Q_seq, R_seq, rx_seq, ru_seq, X0_VAL, N)
        P = cost.eval_q_mat({})
        q = cost.eval_q_vec({})
        c = cost.eval_c({})

        # Dummy decision vector mapping
        # Let's use some arbitrary deterministic values to evaluate the scalar cost
        X_seq = jnp.arange(N * NX).reshape(N, NX) * 0.1
        U_seq = jnp.arange(N * NU).reshape(N, NU) * -0.1
        z = jnp.concatenate([X_seq.reshape(-1), U_seq.reshape(-1)])

        # 1. Cost via our matrix builders
        cost_matrix = 0.5 * z.T @ P @ z + q.T @ z + c

        # 2. Cost via explicit step-by-step for-loop
        x_traj = jnp.vstack([X0_VAL, X_seq])
        cost_loop = 0.0
        for k in range(N):
            err_x = x_traj[k] - rx_seq[k]
            err_u = U_seq[k] - ru_seq[k]
            cost_loop += err_x.T @ Q_seq[k] @ err_x + err_u.T @ R_seq[k] @ err_u
        
        err_x_N = x_traj[N] - rx_seq[N]
        cost_loop += err_x_N.T @ Q_seq[N] @ err_x_N

        assert jnp.isclose(cost_matrix, cost_loop)

    def test_output_tracking_loop_equivalence_random(self):
        """Blockwise (P, q, c) must match an explicit per-step accumulation.

        Complements ``test_output_tracking_loop_equivalence`` with dense
        random C/D and a full (non-diagonal) PSD weight, which exercises
        the state/input cross blocks.
        """
        NY = 3
        rng = np.random.default_rng(0)
        C_seq = jnp.array(rng.normal(size=(N + 1, NY, NX)))
        D_seq = jnp.array(rng.normal(size=(N, NY, NU)))
        r_seq = jnp.array(rng.normal(size=(N + 1, NY)))
        # Symmetric PSD weights, so the quadratic form is a genuine norm.
        W = rng.normal(size=(N + 1, NY, NY))
        Q_seq = jnp.array(np.einsum("tij,tkj->tik", W, W) + np.eye(NY))

        P, q, c = build_output_tracking(C_seq, D_seq, r_seq, Q_seq, X0_VAL, N)

        X_seq = jnp.arange(N * NX).reshape(N, NX) * 0.1
        U_seq = jnp.arange(N * NU).reshape(N, NU) * -0.1
        z = jnp.concatenate([X_seq.reshape(-1), U_seq.reshape(-1)])
        cost_matrix = 0.5 * z.T @ P @ z + q.T @ z + c

        x_traj = jnp.vstack([X0_VAL, X_seq])
        cost_loop = 0.0
        for k in range(N):
            e = C_seq[k] @ x_traj[k] + D_seq[k] @ U_seq[k] - r_seq[k]
            cost_loop += e.T @ Q_seq[k] @ e
        e_N = C_seq[N] @ x_traj[N] - r_seq[N]
        cost_loop += e_N.T @ Q_seq[N] @ e_N

        assert jnp.isclose(cost_matrix, cost_loop)

    def test_split_builders_match_combined(self):
        """The per-field builders must agree with the combined tuple builders."""
        from bpmpc.mpc.helpers._src.costs import (
            build_state_tracking_mat, build_state_tracking_vec,
            build_state_tracking_const,
            build_output_tracking_mat, build_output_tracking_vec,
            build_output_tracking_const,
        )
        NY = 3
        rng = np.random.default_rng(7)
        Q_x = jnp.array(rng.normal(size=(N + 1, NX, NX)))
        R_u = jnp.array(rng.normal(size=(N, NU, NU)))
        rx = jnp.array(rng.normal(size=(N + 1, NX)))
        ru = jnp.array(rng.normal(size=(N, NU)))

        P, q, c = build_state_tracking(Q_x, R_u, rx, ru, X0_VAL, N)
        assert jnp.array_equal(P.todense(), build_state_tracking_mat(Q_x, R_u, N).todense())
        assert jnp.array_equal(q, build_state_tracking_vec(Q_x, R_u, rx, ru, N))
        assert jnp.array_equal(
            c, build_state_tracking_const(Q_x, R_u, rx, ru, X0_VAL, N)
        )

        C_seq = jnp.array(rng.normal(size=(N + 1, NY, NX)))
        D_seq = jnp.array(rng.normal(size=(N, NY, NU)))
        r_seq = jnp.array(rng.normal(size=(N + 1, NY)))
        Q_y = jnp.array(rng.normal(size=(N + 1, NY, NY)))

        P, q, c = build_output_tracking(C_seq, D_seq, r_seq, Q_y, X0_VAL, N)
        assert jnp.array_equal(
            P.todense(), build_output_tracking_mat(C_seq, D_seq, Q_y, N).todense()
        )
        assert jnp.array_equal(
            q, build_output_tracking_vec(C_seq, D_seq, r_seq, Q_y, X0_VAL, N)
        )
        assert jnp.array_equal(
            c, build_output_tracking_const(C_seq, r_seq, Q_y, X0_VAL)
        )

    def test_output_tracking_builder(self):
        NY = 3
        C_seq = jnp.zeros((N + 1, NY, NX))
        D_seq = jnp.zeros((N, NY, NU))
        r_seq = jnp.zeros((N + 1, NY))
        Q_seq = jnp.broadcast_to(jnp.eye(NY), (N + 1, NY, NY))

        P, q, c = build_output_tracking(C_seq, D_seq, r_seq, Q_seq, X0_VAL, N)

        assert P.shape == (N_VAR, N_VAR)
        assert q.shape == (N_VAR,)
        assert c.shape == ()

    def test_output_tracking_factory(self):
        NY = 3
        v_C = Variable("C", (N + 1, NY, NX))
        D_seq = jnp.zeros((N, NY, NU))
        r_seq = jnp.zeros((N + 1, NY))
        Q_seq = jnp.broadcast_to(jnp.eye(NY), (N + 1, NY, NY))
        
        v_x0 = Variable("x0", (NX,))

        cost = output_tracking_cost(v_C, D_seq, r_seq, Q_seq, v_x0, N)

        assert "C" in cost.v_in_q_mat
        assert "x0" in cost.v_in_q_vec

        v_dict = {
            "C": jnp.zeros((N + 1, NY, NX)),
            "x0": X0_VAL,
        }

        q_mat = cost.eval_q_mat(v_dict)
        assert q_mat.shape == (N_VAR, N_VAR)

    def test_output_tracking_auto_tiling_equivalence(self):
        """Verify passing constant output matrices yields the exact same cost as passing broadcasted sequences."""
        NY = 3
        v_x0 = Variable("x0", (NX,))
        C_mat = jnp.ones((NY, NX)) * 0.5
        D_mat = jnp.ones((NY, NU)) * -0.5
        r_vec = jnp.ones(NY) * 2.0
        Q_mat = jnp.eye(NY)

        # 1. Provide single constant elements
        cost_auto = output_tracking_cost(C_mat, D_mat, r_vec, Q_mat, v_x0, N)
        
        # 2. Provide explicitly tiled sequences
        C_seq = jnp.broadcast_to(C_mat, (N + 1, NY, NX))
        D_seq = jnp.broadcast_to(D_mat, (N, NY, NU))
        r_seq = jnp.broadcast_to(r_vec, (N + 1, NY))
        Q_seq = jnp.broadcast_to(Q_mat, (N + 1, NY, NY))
        cost_seq = output_tracking_cost(C_seq, D_seq, r_seq, Q_seq, v_x0, N)

        vars_dict = {"x0": X0_VAL}
        
        assert jnp.allclose(cost_auto.eval_q_mat(vars_dict), cost_seq.eval_q_mat(vars_dict))
        assert jnp.allclose(cost_auto.eval_q_vec(vars_dict), cost_seq.eval_q_vec(vars_dict))
        assert jnp.allclose(cost_auto.eval_c(vars_dict), cost_seq.eval_c(vars_dict))

    def test_output_tracking_loop_equivalence(self):
        """Verify the output block matrix formulation matches a standard for-loop accumulation."""
        NY = 3
        
        # Setup time-varying sequences
        C_seq = jnp.array([jnp.ones((NY, NX)) * (i + 1) for i in range(N + 1)])
        D_seq = jnp.array([jnp.ones((NY, NU)) * (i + 1) for i in range(N)])
        r_seq = jnp.array([jnp.ones(NY) * i for i in range(N + 1)])
        Q_seq = jnp.array([jnp.eye(NY) * (i + 1) for i in range(N + 1)])

        cost = output_tracking_cost(C_seq, D_seq, r_seq, Q_seq, X0_VAL, N)
        P = cost.eval_q_mat({})
        q = cost.eval_q_vec({})
        c = cost.eval_c({})

        # Dummy decision vector mapping
        X_seq = jnp.arange(N * NX).reshape(N, NX) * 0.2
        U_seq = jnp.arange(N * NU).reshape(N, NU) * -0.2
        z = jnp.concatenate([X_seq.reshape(-1), U_seq.reshape(-1)])

        # 1. Cost via our matrix builders
        cost_matrix = 0.5 * z.T @ P @ z + q.T @ z + c

        # 2. Cost via explicit step-by-step for-loop
        x_traj = jnp.vstack([X0_VAL, X_seq])
        cost_loop = 0.0
        for k in range(N):
            err_y = C_seq[k] @ x_traj[k] + D_seq[k] @ U_seq[k] - r_seq[k]
            cost_loop += err_y.T @ Q_seq[k] @ err_y
            
        err_y_N = C_seq[N] @ x_traj[N] - r_seq[N]
        cost_loop += err_y_N.T @ Q_seq[N] @ err_y_N

        assert jnp.isclose(cost_matrix, cost_loop)


# ======================================================================
# Integration: full problem assembly
# ======================================================================

class TestHelperIntegration:
    """Verify helpers compose into a valid MPCProblem."""

    def test_full_assembly(self):
        from bpmpc.mpc import MPCProblem

        v_x0 = Variable("x0", (NX,))

        dynamics = lti_dynamics(A_LTI, B_LTI, v_x0, N)
        
        bounds = box_bounds(
            horizon=N, x_min=X_MIN, x_max=X_MAX, u_min=U_MIN, u_max=U_MAX
        ).add_slack(w_quad=10.0)

        Q = jnp.eye(NX)
        R = 0.1 * jnp.eye(NU)
        r_x = jnp.zeros(NX)
        r_u = jnp.zeros(NU)
        cost = state_tracking_cost(Q, R, r_x, r_u, v_x0, N)

        prob = MPCProblem(
            costs=[cost],
            constraints=[dynamics, bounds],
            # slow_vars=[],
            # fast_vars={"x0"},
            mode="dense",
        )

        assert prob.n_var == N_VAR
        assert prob.n_dec == N_VAR + N_INEQ  # Slack variables added!
        assert prob.n_eq == N_EQ
        assert prob.n_ineq == N_INEQ + N_INEQ # Slack non-negativity added

        prepared = prob._assembler.base_qp
        assert prepared.P.shape == (prob.n_dec, prob.n_dec)

    def test_slack_integration_solve(self):
        """Verifies solving an initially infeasible problem with explicitly separated state and input bounds."""
        from bpmpc.mpc import MPCProblem
        from jaxsparrow import setup_dense_solver

        jax.config.update("jax_enable_x64", True)

        v_x0 = Variable("x0", (NX,))

        # 1. State bounds ONLY, slacked. (We infer inputs are unbound using the `n_u` kwarg).
        bounds_x = box_bounds(
            horizon=N, x_min=X_MIN, x_max=X_MAX, n_u=NU
        ).add_slack(w_quad=1e4, w_lin=100.0)

        # 2. Input bounds ONLY, strict (no slack).
        bounds_u = box_bounds(
            horizon=N, u_min=U_MIN, u_max=U_MAX, n_x=NX
        )

        dynamics = lti_dynamics(A_LTI, B_LTI, v_x0, N)

        Q = jnp.eye(NX)
        R = 0.1 * jnp.eye(NU)
        r_x = jnp.zeros(NX)
        r_u = jnp.zeros(NU)
        cost = state_tracking_cost(Q, R, r_x, r_u, v_x0, N)

        prob = MPCProblem(
            costs=[cost],
            constraints=[dynamics, bounds_x, bounds_u],
            mode="dense",
        )

        # create solver
        jxp = setup_dense_solver(
            n_var=prob.n_dec,
            n_ineq=prob.n_ineq,
            n_eq=prob.n_eq,
            fixed_elements=prob.fixed_elements,
        )

        n_ineq_input = NU*2*N
        n_ineq_state = NX*2*N
        n_ineq_slack = n_ineq_state

        # Dimensions check
        # box_bounds evaluates across the entire space regardless of the omissions
        assert prob.n_var == N_VAR
        assert prob.n_eq == N_EQ
        assert prob.n_ineq == n_ineq_input + n_ineq_state + n_ineq_slack
        assert prob.n_dec == N_VAR + n_ineq_slack  # Only bounds_x adds slack variables

        solver = prob.with_solver(jxp)

        # Setup an incredibly infeasible initial state (requires slack because max limit is 5.0)
        x0_infeasible = jnp.array([20.0, 20.0])
        sol = solver.solve({"x0": x0_infeasible})

        # Check solver flat output. If 'z' isn't explicitly exposed depending on your MPCSolver API, 
        # this will safely skip rather than randomly crashing your test suite.
        z_opt = sol["x"]

        z_primal = z_opt[:N_VAR]
        slacks = z_opt[N_VAR:]

        # 1. Check that inputs are physically restricted to strict bounds despite infeasibility
        u_opt = z_primal[N * NX:]
        assert jnp.all(u_opt >= U_MIN[0] - 1e-4)
        assert jnp.all(u_opt <= U_MAX[0] + 1e-4)

        # 2. Check that the solver successfully tapped into the slack variables
        assert jnp.any(slacks > 1e-4), "Solver failed to utilize slacks for an infeasible trajectory."
        
        # 3. Prove that the states indeed crossed the absolute physical threshold natively
        lhs_x = bounds_x.eval_lhs({})
        rhs_x = bounds_x.eval_rhs({})
        
        raw_violation = (lhs_x @ z_primal) - rhs_x
        assert jnp.any(raw_violation > 1e-4), "State was mathematically feasible without slacks! (x0 wasn't infeasible enough)"

# ======================================================================
# Sparse builder mode
# ======================================================================

class TestSparseBuilders:
    """Matrix builders return a BCOO by default, equal to the ``sparse=False`` array."""

    @staticmethod
    def _cases():
        NY = 3
        rng = np.random.default_rng(11)
        Q_x = jnp.array(rng.normal(size=(N + 1, NX, NX)))
        R_u = jnp.array(rng.normal(size=(N, NU, NU)))
        C_seq = jnp.array(rng.normal(size=(N + 1, NY, NX)))
        D_seq = jnp.array(rng.normal(size=(N, NY, NU)))
        Q_y = jnp.array(rng.normal(size=(N + 1, NY, NY)))
        rx, ru = jnp.zeros((N + 1, NX)), jnp.zeros((N, NU))
        r_y = jnp.zeros((N + 1, NY))
        return {
            "lti_lhs": lambda **kw: build_lti_lhs(A_LTI, B_LTI, N, **kw),
            "ltv_lhs": lambda **kw: build_ltv_lhs(A_LTV, B_LTV, N, **kw),
            "box_lhs": lambda **kw: build_box_lhs(NX, NU, N, **kw),
            "state_P": lambda **kw: build_state_tracking(
                Q_x, R_u, rx, ru, X0_VAL, N, **kw)[0],
            "output_P": lambda **kw: build_output_tracking(
                C_seq, D_seq, r_y, Q_y, X0_VAL, N, **kw)[0],
        }

    @pytest.mark.parametrize("name", ["lti_lhs", "ltv_lhs", "box_lhs", "state_P", "output_P"])
    def test_sparse_matches_dense(self, name):
        from jax.experimental.sparse import BCOO
        build = self._cases()[name]
        dense, sparse = build(sparse=False), build()
        assert not isinstance(dense, BCOO)
        assert isinstance(sparse, BCOO)
        np.testing.assert_array_equal(np.asarray(sparse.todense()), np.asarray(dense))

    def test_sparse_builder_inside_parametric_cost(self):
        """A lambda returning the sparse builder output gives the same QP matrix."""
        from bpmpc.mpc import Cost
        v_R = Variable("R", (NU, NU))
        Q_x = jnp.broadcast_to(jnp.eye(NX), (N + 1, NX, NX))

        def make(sparse):
            def q_mat(v):
                R = jnp.broadcast_to(v["R"], (N, NU, NU))
                return build_state_tracking(
                    Q_x, R, jnp.zeros((N + 1, NX)), jnp.zeros((N, NU)),
                    X0_VAL, N, sparse=sparse)[0]
            return Cost(q_mat=q_mat, v_in_q_mat={"R": v_R})

        sparse, dense = make(True), make(False)
        assert sparse._q_mat_nz is not None
        v = {"R": 3.0 * jnp.eye(NU)}
        np.testing.assert_allclose(sparse.eval_q_mat(v), dense.eval_q_mat(v))
