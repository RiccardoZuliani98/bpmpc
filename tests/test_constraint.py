"""Tests for Constraint descriptor."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from bpmpc.variable import Variable
from bpmpc.mpc import Constraint, SlackSpec


def test_constant_constraint():
    cst = Constraint(
        "equality",
        lhs=lambda _: jnp.ones((2, 3)),
        rhs=lambda _: jnp.array([5.0, 6.0]),
    )
    assert cst.is_equality
    assert not cst.is_parametric
    assert cst.n_cst == 2
    assert cst.n_var == 3
    assert not cst.is_slacked

    np.testing.assert_array_equal(cst.eval_lhs(), np.ones((2, 3)))
    np.testing.assert_array_equal(cst.eval_rhs(), np.array([5.0, 6.0]))


def test_parametric_constraint():
    v_A = Variable("A", (2, 3))
    cst = Constraint(
        "inequality",
        lhs=lambda v: v["A"],
        v_in_lhs={"A": v_A},
        rhs=lambda _: jnp.zeros(2),
    )
    assert not cst.is_equality
    assert cst.is_parametric
    assert cst.is_parametric_lhs
    assert not cst.is_parametric_rhs

    A_val = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(cst.eval_lhs({"A": A_val}), A_val)


def test_constraint_validation_shapes():
    # LHS not 2D
    with pytest.raises(ValueError, match="LHS must return a 2-D array"):
        Constraint("equality", lhs=lambda _: jnp.ones(3), rhs=lambda _: jnp.zeros(3))
    
    # RHS not 1D
    with pytest.raises(ValueError, match="RHS must return a 1-D array"):
        Constraint("equality", lhs=lambda _: jnp.ones((2, 3)), rhs=lambda _: jnp.zeros((2, 1)))

    # Row count mismatch
    with pytest.raises(ValueError, match="LHS has 2 rows but RHS has 3 elements"):
        Constraint("equality", lhs=lambda _: jnp.ones((2, 3)), rhs=lambda _: jnp.zeros(3))


def test_constraint_slack_integration():
    slack = SlackSpec.slack_all(2, w_quad=1.0)
    
    # Slacking equality is forbidden
    with pytest.raises(ValueError, match="Slack variables are only supported on inequality"):
        Constraint("equality", lhs=lambda _: jnp.ones((2, 3)), rhs=lambda _: jnp.zeros(2), slack=slack)

    # Slack dimension mismatch
    with pytest.raises(ValueError, match="SlackSpec has n_cst=2 but the constraint has n_cst=3"):
        Constraint("inequality", lhs=lambda _: jnp.ones((3, 3)), rhs=lambda _: jnp.zeros(3), slack=slack)

    # Valid slacked constraint
    cst = Constraint("inequality", lhs=lambda _: jnp.ones((2, 3)), rhs=lambda _: jnp.zeros(2), slack=slack)
    assert cst.is_slacked
    assert cst.slack.n_slack == 2


def test_constraint_addition():
    c1 = Constraint("equality", lhs=lambda _: jnp.ones((1, 2)), rhs=lambda _: jnp.array([1.0]))
    c2 = Constraint("equality", lhs=lambda _: jnp.ones((2, 2)) * 2, rhs=lambda _: jnp.array([2.0, 3.0]))
    
    c_sum = c1 + c2
    assert c_sum.n_cst == 3
    assert c_sum.n_var == 2
    
    np.testing.assert_array_equal(c_sum.eval_lhs(), np.array([[1, 1], [2, 2], [2, 2]]))
    np.testing.assert_array_equal(c_sum.eval_rhs(), np.array([1.0, 2.0, 3.0]))

    # Type mismatch
    c3 = Constraint("inequality", lhs=lambda _: jnp.ones((1, 2)), rhs=lambda _: jnp.zeros(1))
    with pytest.raises(ValueError, match="Cannot add constraints of different types"):
        _ = c1 + c3

    # Var mismatch
    c4 = Constraint("equality", lhs=lambda _: jnp.ones((1, 3)), rhs=lambda _: jnp.zeros(1))
    with pytest.raises(ValueError, match="Cannot add constraints with different n_var"):
        _ = c1 + c4

def test_constraint_slack_modifier():
    # Create a base inequality constraint without slacks
    cst_ineq = Constraint(
        "inequality",
        lhs=jnp.ones((3, 2)),
        rhs=jnp.zeros(3)
    )
    assert not cst_ineq.is_slacked
    
    # Test slacking all rows
    cst_slacked_all = cst_ineq.add_slack(w_quad=10.0, w_lin=1.0)
    assert cst_slacked_all.is_slacked
    assert cst_slacked_all.slack.n_slack == 3
    np.testing.assert_array_equal(cst_slacked_all.slack.w_quad_array, np.array([10.0, 10.0, 10.0]))
    np.testing.assert_array_equal(cst_slacked_all.slack.w_lin_array, np.array([1.0, 1.0, 1.0]))
    
    # Test slacking specific rows
    cst_slacked_partial = cst_ineq.add_slack(rows=[0, 2], w_quad=5.0)
    assert cst_slacked_partial.is_slacked
    assert cst_slacked_partial.slack.n_slack == 2
    np.testing.assert_array_equal(cst_slacked_partial.slack.rows_array, np.array([1, 0, 1]))
    np.testing.assert_array_equal(cst_slacked_partial.slack.w_quad_array, np.array([5.0, 5.0]))
    
    # Verify original constraint is unchanged (immutability)
    assert not cst_ineq.is_slacked

    # Test error on equality constraint
    cst_eq = Constraint(
        "equality",
        lhs=lambda _: jnp.ones((2, 2)),
        rhs=lambda _: jnp.zeros(2)
    )
    with pytest.raises(ValueError, match="only supported on inequality constraints"):
        cst_eq.add_slack()