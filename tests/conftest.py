"""Shared fixtures for MPCProblem tests.

Uses a small double-integrator MPC as the reference problem.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from bpmpc.variable import Variable
from bpmpc.mpc import Cost, Constraint, SlackSpec, MPCProblem

NX, NU, N, DT = 2, 1, 20, 0.1

A_D = jnp.array([[1.0, DT], [0.0, 1.0]])
B_D = jnp.array([[0.5 * DT**2], [DT]])

N_VAR    = N * (NX + NU) + NX          # 17
N_EQ_DYN = N * NX                      # 10
N_EQ     = N_EQ_DYN + NX               # 12 (dynamics + IC)
N_INEQ   = 2 * (NX + NU) * N + 2 * NX  # 34


@pytest.fixture
def v_p():
    return Variable("p", (N_VAR,))


@pytest.fixture
def v_x0():
    return Variable("x0", (NX,))


@pytest.fixture
def cost_obj(v_p):
    return Cost(
        q_mat=lambda v: jnp.diag(v["p"] ** 2 + 1e-8),
        v_in_q_mat={"p": v_p},
    )


@pytest.fixture
def dynamics_obj():
    A_eq = jnp.zeros((N_EQ_DYN, N_VAR))
    for k in range(N):
        r, cx = k * NX, k * (NX + NU)
        cu, cx1 = cx + NX, (k + 1) * (NX + NU)
        A_eq = A_eq.at[r:r+NX, cx:cx+NX].set(A_D)
        A_eq = A_eq.at[r:r+NX, cu:cu+NU].set(B_D)
        A_eq = A_eq.at[r:r+NX, cx1:cx1+NX].set(-jnp.eye(NX))
    rhs = jnp.zeros(N_EQ_DYN)
    return Constraint("equality", lhs=lambda _: A_eq, rhs=lambda _: rhs)


@pytest.fixture
def ic_obj(v_x0):
    lhs = jnp.zeros((NX, N_VAR)).at[:NX, :NX].set(jnp.eye(NX))
    return Constraint(
        "equality",
        lhs=lambda _: lhs, 
        rhs=lambda v: v["x0"], 
        v_in_rhs={"x0": v_x0}
    )


@pytest.fixture
def bounds_obj():
    x_lb, x_ub = jnp.array([-10., -5.]), jnp.array([10., 5.])
    u_lb, u_ub = jnp.array([-1.]), jnp.array([1.])
    G = jnp.zeros((N_INEQ, N_VAR))
    h = jnp.zeros(N_INEQ)
    row = 0
    for k in range(N):
        cx, cu = k * (NX + NU), k * (NX + NU) + NX
        G = G.at[row:row+NX, cx:cx+NX].set(jnp.eye(NX));    h = h.at[row:row+NX].set(x_ub);  row += NX
        G = G.at[row:row+NX, cx:cx+NX].set(-jnp.eye(NX));   h = h.at[row:row+NX].set(-x_lb); row += NX
        G = G.at[row:row+NU, cu:cu+NU].set(jnp.eye(NU));    h = h.at[row:row+NU].set(u_ub);  row += NU
        G = G.at[row:row+NU, cu:cu+NU].set(-jnp.eye(NU));   h = h.at[row:row+NU].set(-u_lb); row += NU
    cxN = N * (NX + NU)
    G = G.at[row:row+NX, cxN:cxN+NX].set(jnp.eye(NX));  h = h.at[row:row+NX].set(x_ub);  row += NX
    G = G.at[row:row+NX, cxN:cxN+NX].set(-jnp.eye(NX)); h = h.at[row:row+NX].set(-x_lb); row += NX
    slack = SlackSpec.slack_all(N_INEQ, w_quad=10.0, w_lin=5.0)
    return Constraint("inequality", lhs=lambda _: G, rhs=lambda _: h, slack=slack)


@pytest.fixture(params=["dense", "sparse"])
def mode(request):
    """Yields 'dense' and 'sparse' to parametrize tests."""
    return request.param

@pytest.fixture
def problem(cost_obj, dynamics_obj, ic_obj, bounds_obj, mode):
    """A generic parametrized problem that tests both dense and sparse assembly."""
    return MPCProblem(
        costs=[cost_obj],
        constraints=[dynamics_obj, ic_obj, bounds_obj],
        slow_vars=["p"],
        mode=mode,
    )

def dummy_solver(P, q, A, b, G, h, **kwargs):
    """Dummy solver that expects identical API regardless of sparsity."""
    return {"x": jnp.zeros(P.shape[1])}