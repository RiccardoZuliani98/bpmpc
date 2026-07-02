"""Tests for Cost descriptor."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from bpmpc.variable import Variable
from bpmpc.mpc import Cost


def test_constant_cost():
    cost = Cost(
        q_mat=lambda _: jnp.eye(3),
        q_vec=lambda _: jnp.ones(3),
        c=lambda _: jnp.array(5.0),
    )
    assert cost.n_var == 3
    assert not cost.is_parametric
    assert not cost.is_parametric_q_mat
    assert not cost.is_parametric_q_vec
    assert not cost.is_parametric_c

    # Evaluation handles empty dict cleanly
    np.testing.assert_array_equal(cost.eval_q_mat(), np.eye(3))
    np.testing.assert_array_equal(cost.eval_q_vec(), np.ones(3))
    np.testing.assert_array_equal(cost.eval_c(), 5.0)


def test_parametric_cost():
    v_p = Variable("p", (3,))
    v_w = Variable("w", ())

    cost = Cost(
        q_mat=lambda v: jnp.diag(v["p"]),
        v_in_q_mat={"p": v_p},
        c=lambda v: v["w"] * 2.0,
        v_in_c={"w": v_w},
    )
    assert cost.n_var == 3
    assert cost.is_parametric
    assert cost.is_parametric_q_mat
    assert not cost.is_parametric_q_vec
    assert cost.is_parametric_c

    p_val = jnp.array([1.0, 2.0, 3.0])
    w_val = jnp.array(4.0)

    np.testing.assert_array_equal(cost.eval_q_mat({"p": p_val}), np.diag([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(cost.eval_q_vec(), np.zeros(3))  # Defaults to 0
    np.testing.assert_array_equal(cost.eval_c({"w": w_val}), 8.0)


def test_cost_validation_shapes():
    # q_mat not 2D
    with pytest.raises(ValueError, match="q_mat must return a 2-D array"):
        Cost(q_mat=lambda _: jnp.ones(3))

    # q_mat not square
    with pytest.raises(ValueError, match="q_mat must be square"):
        Cost(q_mat=lambda _: jnp.ones((3, 4)))

    # q_vec wrong length
    with pytest.raises(ValueError, match="q_vec has length 4 but q_mat is 3x3"):
        Cost(q_mat=lambda _: jnp.eye(3), q_vec=lambda _: jnp.ones(4))

    # c not scalar
    with pytest.raises(ValueError, match="c must return a scalar"):
        Cost(q_mat=lambda _: jnp.eye(3), c=lambda _: jnp.ones(1))


def test_cost_validation_shared_variables():
    v1 = Variable("shared", (2,))
    v2 = Variable("shared", (3,))  # Same name, different shape
    
    with pytest.raises(ValueError, match="Shared key 'shared'"):
        Cost(
            q_mat=lambda v: jnp.eye(3),
            v_in_q_mat={"shared": v1},
            q_vec=lambda v: jnp.zeros(3),
            v_in_q_vec={"shared": v2},
        )


def test_cost_addition():
    v_p = Variable("p", (2,))
    
    c1 = Cost(
        q_mat=lambda _: jnp.eye(2),
        q_vec=lambda v: v["p"],
        v_in_q_vec={"p": v_p},
        c=lambda _: jnp.array(1.0)
    )
    c2 = Cost(
        q_mat=lambda _: jnp.eye(2) * 2.0,
        c=lambda _: jnp.array(3.0)
    )
    
    c_sum = c1 + c2
    assert c_sum.n_var == 2
    assert c_sum.is_parametric
    
    p_val = jnp.array([5.0, 6.0])
    np.testing.assert_array_equal(c_sum.eval_q_mat(), np.eye(2) * 3.0)
    np.testing.assert_array_equal(c_sum.eval_q_vec({"p": p_val}), np.array([5.0, 6.0]))
    np.testing.assert_array_equal(c_sum.eval_c(), 4.0)

    # Addition fails if n_var mismatched
    c3 = Cost(q_mat=lambda _: jnp.eye(3))
    with pytest.raises(ValueError, match="Cannot add costs with different n_var"):
        _ = c1 + c3