"""Tests for SlackSpec."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from bpmpc.mpc import SlackSpec


def test_slack_all_and_none():
    s_all = SlackSpec.slack_all(3, w_quad=10.0, w_lin=5.0)
    assert s_all.n_cst == 3
    assert s_all.n_slack == 3
    assert s_all.rows == (1, 1, 1)
    assert s_all.w_quad == (10.0, 10.0, 10.0)
    assert s_all.w_lin == (5.0, 5.0, 5.0)
    assert not s_all.is_empty

    s_none = SlackSpec.slack_none(3)
    assert s_none.n_cst == 3
    assert s_none.n_slack == 0
    assert s_none.is_empty
    assert s_none.rows == (0, 0, 0)


def test_slack_rows():
    s = SlackSpec.slack_rows(4, rows=[1, 3], w_quad=[1.0, 2.0], w_lin=0.5)
    
    assert s.n_cst == 4
    assert s.n_slack == 2
    assert s.rows == (0, 1, 0, 1)
    assert s.w_quad == (1.0, 2.0)
    assert s.w_lin == (0.5, 0.5)  # Scalar broadcasted
    assert s.slack_indices == (1, 3)

    np.testing.assert_array_equal(s.rows_array, np.array([0, 1, 0, 1]))


def test_slack_validation_bounds():
    with pytest.raises(ValueError, match="Slack row index 5 is out of range"):
        SlackSpec.slack_rows(3, rows=[5])

    with pytest.raises(ValueError, match="Duplicate row indices"):
        SlackSpec.slack_rows(3, rows=[1, 1])


def test_slack_validation_weights():
    # Mismatched sequence length
    with pytest.raises(ValueError, match="w_quad has length 3 but n_slack=2"):
        SlackSpec.slack_rows(4, rows=[0, 1], w_quad=[1.0, 2.0, 3.0])

    # Negative weights
    with pytest.raises(ValueError, match="penalty weights must be non-negative"):
        SlackSpec.slack_all(2, w_quad=-1.0)


def test_slack_addition():
    s1 = SlackSpec.slack_rows(2, rows=[0], w_quad=1.0)  # [1, 0]
    s2 = SlackSpec.slack_rows(3, rows=[1, 2], w_quad=2.0)  # [0, 1, 1]
    
    s_sum = s1 + s2
    
    assert s_sum.n_cst == 5
    assert s_sum.n_slack == 3
    assert s_sum.rows == (1, 0, 0, 1, 1)
    assert s_sum.w_quad == (1.0, 2.0, 2.0)
    assert s_sum.slack_indices == (0, 3, 4)