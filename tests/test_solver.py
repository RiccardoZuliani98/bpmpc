"""Tests for attaching solvers to MPCProblems."""

from __future__ import annotations

import pytest

import numpy as np
import jax.numpy as jnp

from jax.experimental.sparse import BCOO

from bpmpc.mpc import MPCSolver, MPCProblem
from conftest import N_VAR, N_EQ, N_INEQ, NX, dummy_solver

def to_dense(mat):
    """Helper to unify dense and BCOO matrices for numpy testing."""
    return np.asarray(mat.todense() if isinstance(mat, BCOO) else mat)

def test_returns_mpc_solver(problem):
    mpc = problem.with_solver(dummy_solver)
    assert isinstance(mpc, MPCSolver)


def test_solver_metadata(problem):
    mpc = problem.with_solver(dummy_solver)
    assert mpc.n_var == N_VAR
    assert mpc.n_dec == N_VAR + N_INEQ
    assert mpc.n_eq == N_EQ
    assert mpc.n_ineq == 2*N_INEQ
    assert mpc.mode == "dense" or mpc.mode == "sparse"
    assert "p" in mpc.all_vars
    assert "x0" in mpc.all_vars


def test_prepare_shapes_are_fully_expanded(problem):
    mpc = problem.with_solver(dummy_solver)
    prepared = mpc.prepare({"p": jnp.ones(N_VAR)})
    
    # In the new code base, slacks are folded dynamically into the base matrix 
    # making `prepared` naturally returned fully sized!
    n_dec = N_VAR + N_INEQ
    assert prepared.P.shape == (n_dec, n_dec)
    assert prepared.A.shape == (N_EQ, n_dec)


def test_solve_matches_manual_assembly(problem):
    mpc = problem.with_solver(dummy_solver)
    p_val  = {"p": jnp.ones(N_VAR) * 0.5}
    x0_val = {"x0": jnp.array([1.0, -0.5])}

    # Solve flow
    prepared = mpc.prepare(p_val)
    result = mpc.solve_with_prepared(fast_v=x0_val, warmstart=None, prepared_qp=prepared)

    # Assemble manually
    qp_manual = problem.prepare(p_val)

    for key in ("P", "q", "A", "b", "G", "h"):
        np.testing.assert_allclose(
            # np.asarray(result[key]),
            to_dense(getattr(prepared, key)),
            to_dense(getattr(qp_manual, key)),
            rtol=1e-6, err_msg=f"{key} mismatch solver vs assemble",
        )

@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_output_partition(cost_obj, dynamics_obj, ic_obj, bounds_obj, mode):
    prob = MPCProblem(
        costs=[cost_obj], constraints=[dynamics_obj, ic_obj, bounds_obj],
        outputs={"ctrl": slice(0, 1), "state": (1, 3)},
        mode=mode
    )
    mpc = prob.with_solver(dummy_solver)
    
    # Solve directly treating all as fast parameters
    sol = mpc.solve({"p": jnp.ones(N_VAR), "x0": jnp.zeros(NX)}, warmstart=None)
    
    assert "ctrl" in sol["x"]
    assert sol["x"]["ctrl"].shape == (1,)
    assert sol["x"]["state"].shape == (2,)


def test_multiple_solvers_independent(problem):
    """Attaching different solvers to the same problem is fine."""
    calls = []

    def solver_a(*args, **kwargs):
        calls.append("a")
        return {"x": jnp.zeros(10)}

    def solver_b(*args, **kwargs):
        calls.append("b")
        return {"x": jnp.zeros(10)}

    mpc_a = problem.with_solver(solver_a)
    mpc_b = problem.with_solver(solver_b)

    prepared = mpc_a.prepare({"p": jnp.ones(N_VAR)})
    x0 = {"x0": jnp.zeros(NX)}
    
    mpc_a.solve_with_prepared(fast_v=x0, warmstart=None, prepared_qp=prepared)
    mpc_b.solve_with_prepared(fast_v=x0, warmstart=None, prepared_qp=prepared)
    assert calls == ["a", "b"]