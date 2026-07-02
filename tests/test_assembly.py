"""Tests for dense assembly (prepare / solve_from)."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from conftest import N_VAR, N_EQ, N_EQ_DYN, N_INEQ, NX

def to_dense(mat):
    """Helper to unify dense and BCOO matrices for numpy testing."""
    return np.asarray(mat.todense() if isinstance(mat, BCOO) else mat)

def test_assembly_shapes(problem):
    p_val  = {"p": jnp.ones(N_VAR) * 0.5}
    x0_val = {"x0": jnp.array([1.0, 0.0])}
    
    prepared = problem.prepare(p_val)
    qp = problem.solve_from(prepared, x0_val)
    
    n_dec = N_VAR + N_INEQ
    assert qp.P.shape == (n_dec, n_dec)
    assert qp.q.shape == (n_dec,)
    assert qp.A.shape == (N_EQ, n_dec)
    assert qp.b.shape == (N_EQ,)
    assert qp.G.shape == (2*N_INEQ, n_dec)
    assert qp.h.shape == (2*N_INEQ,)


def test_cost_diagonal(problem):
    p_v = jnp.ones(N_VAR) * 0.5
    qp = problem.solve_from(problem.prepare({"p": p_v}), {"x0": jnp.zeros(NX)})
    
    P_diag = jnp.diag(to_dense(qp.P))
    expected = jnp.concatenate([p_v**2 + 1e-8, 20.0 * jnp.ones(N_INEQ)]) # multiply by two to counteract the 0.5
    np.testing.assert_allclose(P_diag, expected, rtol=1e-5)


def test_slack_linear_penalty(problem):
    qp = problem.solve_from(problem.prepare({"p": jnp.ones(N_VAR)}), {"x0": jnp.zeros(NX)})
    q_slack = qp.q[N_VAR:]
    np.testing.assert_allclose(q_slack, 5.0 * jnp.ones(N_INEQ), rtol=1e-5)


def test_initial_condition_in_b(problem):
    x0_val = jnp.array([3.0, -2.0])
    qp = problem.solve_from(problem.prepare({"p": jnp.ones(N_VAR)}), {"x0": x0_val})
    
    b_ic = qp.b[N_EQ_DYN : N_EQ_DYN + NX]
    np.testing.assert_array_equal(b_ic, x0_val)


def test_dynamics_rhs_zero(problem):
    qp = problem.solve_from(problem.prepare({"p": jnp.ones(N_VAR)}), {"x0": jnp.zeros(NX)})
    np.testing.assert_array_equal(qp.b[:N_EQ_DYN], jnp.zeros(N_EQ_DYN))


def test_G_slack_columns(problem):
    qp = problem.solve_from(
        problem.prepare({"p": jnp.ones(N_VAR)}),
        {"x0": jnp.zeros(NX)},
    )
    G_slack = np.asarray(to_dense(qp.G)[:, N_VAR:])
    expected = np.vstack([-np.eye(N_INEQ), -np.eye(N_INEQ)])
    np.testing.assert_array_equal(G_slack, expected)


def test_A_slack_columns_zero(problem):
    qp = problem.solve_from(problem.prepare({"p": jnp.ones(N_VAR)}), {"x0": jnp.zeros(NX)})
    A_slack = np.asarray(to_dense(qp.A)[:, N_VAR:])
    np.testing.assert_array_equal(A_slack, np.zeros((N_EQ, N_INEQ)))


def test_different_p_changes_cost(problem):
    x0 = {"x0": jnp.zeros(NX)}
    qp1 = problem.solve_from(problem.prepare({"p": jnp.ones(N_VAR)}), x0)
    qp2 = problem.solve_from(problem.prepare({"p": 2.0 * jnp.ones(N_VAR)}), x0)
    
    assert not np.allclose(to_dense(qp1.P), to_dense(qp2.P))


def test_different_x0_changes_b_but_leaves_P_alone(problem):
    prepared = problem.prepare({"p": jnp.ones(N_VAR)})
    
    qp1 = problem.solve_from(prepared, {"x0": jnp.array([1.0, 0.0])})
    qp2 = problem.solve_from(prepared, {"x0": jnp.array([0.0, 1.0])})
    
    assert not np.array_equal(np.asarray(qp1.b), np.asarray(qp2.b))
    np.testing.assert_array_equal(to_dense(qp1.P), to_dense(qp2.P))