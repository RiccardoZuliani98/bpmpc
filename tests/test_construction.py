"""Tests for MPCProblem construction, dimensions, and variables."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from bpmpc.mpc import Cost, Constraint, MPCProblem
from conftest import N_VAR, N_EQ, N_INEQ


def test_dimensions_dense(problem):
    p = problem
    assert p.n_var == N_VAR
    assert p.n_eq == N_EQ
    assert p.n_ineq == 2*N_INEQ
    assert p.n_dec == N_VAR + N_INEQ


def test_variable_classification(problem):
    p = problem
    # Ensure variables extracted from terms end up in all_vars properly
    assert "p" in p.all_vars
    assert "x0" in p.all_vars


def test_fixed_elements(problem):
    # 'G' and 'h' are from bounds_obj, which has no parametric dependencies.
    # Therefore, they shouldn't be touched by the solver at run-time.
    fixed = problem.fixed_elements
    assert "G" in fixed
    assert "h" in fixed
    
    # 'P', 'q', 'A', 'b' all depend on parameters or slacks in our fixture
    assert "P" not in fixed
    assert "b" not in fixed


def test_repr(problem):
    r = repr(problem)
    assert "MPCProblem" in r
    assert "dense" in r or "sparse" in r
    assert "slow_vars=['p']" in r


def test_no_slack_dimensions(cost_obj, dynamics_obj, ic_obj):
    """Problem without inequality constraints has no slacks."""
    prob = MPCProblem(
        costs=[cost_obj], constraints=[dynamics_obj, ic_obj],
        slow_vars=["p"], mode="dense",
    )
    # Without inequality slacks, n_dec should strictly equal n_var
    assert prob.n_dec == prob.n_var
    assert prob.n_ineq == 0


def test_constant_cost_and_constraints():
    """Purely constant problem solves instantly from base QP."""
    n = 3
    cost = Cost(q_mat=lambda _: jnp.eye(n))
    cst = Constraint(
        "equality",
        lhs=lambda _: jnp.eye(2, n), 
        rhs=lambda _: jnp.zeros(2)
    )
    prob = MPCProblem(
        costs=[cost], constraints=[cst],
        mode="dense",
    )
    assert prob.n_var == n
    assert prob.n_eq == 2

    prepared = prob.prepare({})
    qp = prob.solve_from(prepared, {})
    np.testing.assert_array_equal(np.asarray(qp.P), np.eye(n))
    np.testing.assert_array_equal(np.asarray(qp.A), np.eye(2, n))