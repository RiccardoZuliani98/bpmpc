"""Tests for input validation when building an MPCProblem."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from bpmpc.mpc import Cost, Constraint, MPCProblem


def test_no_costs(dynamics_obj, ic_obj, bounds_obj):
    with pytest.raises(ValueError, match="At least one cost is required"):
        MPCProblem(costs=[], constraints=[dynamics_obj, ic_obj, bounds_obj])


def test_undeclared_variable_in_slow_vars(cost_obj, dynamics_obj, ic_obj, bounds_obj):
    # slow_vars demands that provided names are actually known to the problem
    with pytest.raises(ValueError, match="slow_vars contains unknown names"):
        MPCProblem(
            costs=[cost_obj], constraints=[dynamics_obj, ic_obj, bounds_obj],
            slow_vars=["p", "missing_var"]
        )


def test_inconsistent_n_var():
    cost_3 = Cost(q_mat=lambda _: jnp.eye(3))
    cost_4 = Cost(q_mat=lambda _: jnp.eye(4))
    cst = Constraint(
        "equality",
        lhs=lambda _: jnp.eye(2, 3), 
        rhs=lambda _: jnp.zeros(2)
    )
    with pytest.raises(ValueError, match="Inconsistent n_var"):
        MPCProblem(costs=[cost_3, cost_4], constraints=[cst])


def test_invalid_outputs_partition(cost_obj, dynamics_obj, ic_obj):
    with pytest.raises(ValueError, match="indices out of range"):
        MPCProblem(
            costs=[cost_obj], constraints=[dynamics_obj, ic_obj],
            outputs={"invalid": slice(0, 1000)}  # Way out of bounds
        )