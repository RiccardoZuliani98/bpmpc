#!/usr/bin/env python3
"""Closed-loop MPC with ``Dynamics`` and ``ClosedLoopSimulator``.

Double integrator, sparse mode.  ``cost_state`` is a 2-vector on the
diagonal of the MPC's state cost, declared slow: one ``prepare`` per
episode, many ``solve`` calls.

The true dynamics carry a small unmodeled drift ``w`` (a constant
disturbance unknown to the controller); the nominal dynamics is the
clean double integrator.  The ``Dynamics`` class combines them via a
straight-through estimator: forward uses the true system, AD routes
through the nominal model.

Two rollouts:

  1. A verbose rollout prints per-step state / control.
  2. A silent rollout is wrapped in ``jax.jvp`` to differentiate the
     trajectory w.r.t. ``cost_state``.  A finite-difference check
     verifies the JVP is correct.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from bpmpc.mpc import (
    Cost, Constraint, MPCProblem,
)
from bpmpc.closed_loop import ClosedLoop, RunLogger, TrajectoryStorage
from bpmpc.mpc.helpers import lti_dynamics, box_bounds
from bpmpc.dynamics import Dynamics
from bpmpc.variable import Variable
from jaxsparrow import setup_sparse_solver


# ======================================================================
# Problem data
# ======================================================================

nx, nu, N   = 2, 1, 50
N_sim       = 80
x0_init     = jnp.array([-3.0, -1.0])
cost_state0 = jnp.array([1.0, 1.0])          # parameter to differentiate w.r.t.
cost_input  = 0.1

A_d = jnp.array([[1.0, 1.0], [0.0, 1.0]])
B_d = jnp.array([[0.0], [1.0]])

# True-system disturbance — unmodeled drift the MPC doesn't know about.
w_true = jnp.array([0.0, 0.02])

x_min = jnp.array([-5.0, -5.0]); x_max = jnp.array([5.0, 5.0])
u_min = jnp.array([-0.5]);        u_max = jnp.array([0.5])

v_x0         = Variable("x0",         (nx,))
v_cost_state = Variable("cost_state", (nx,))


# ======================================================================
# MPC: cost, dynamics (for the optimizer), bounds
# ======================================================================

n_var = N * nx + N * nu

def q_mat(v):
    Q = jnp.diag(v["cost_state"])
    R = cost_input * jnp.eye(nu)
    Q_blk = jnp.kron(jnp.eye(N), Q)
    R_blk = jnp.kron(jnp.eye(N), R)
    return jnp.block([
        [Q_blk,                         jnp.zeros((N * nx, N * nu))],
        [jnp.zeros((N * nu, N * nx)),  R_blk                       ],
    ])

mpc_cost     = Cost(q_mat=q_mat, v_in_q_mat={"cost_state": v_cost_state})
mpc_dynamics = lti_dynamics(A=A_d, B=B_d, x0=v_x0, horizon=N)
mpc_bounds   = box_bounds(x_min=x_min, x_max=x_max,
                          u_min=u_min, u_max=u_max, horizon=N)

problem = MPCProblem(
    costs=[mpc_cost],
    constraints=[mpc_dynamics, mpc_bounds],
    slow_vars=["cost_state"],
    outputs={"u0": slice(N * nx, N * nx + nu)},   # first control
    mode="sparse",
)
print(repr(problem))

sparsity_patterns = problem.sparsity_pattern

n_dec = problem.n_dec
n_eq = problem.n_eq
n_ineq = problem.n_ineq

print(f"  Sparsity — "
      f"P: {sparsity_patterns['P'].nse} nnz / {n_dec * n_dec}, "
      f"A: {sparsity_patterns['A'].nse} nnz / {n_eq * n_dec}, "
      f"G: {sparsity_patterns['G'].nse} nnz / {n_ineq * n_dec}")

solver = setup_sparse_solver(
    n_var=problem.n_dec,
    n_ineq=problem.n_ineq,
    n_eq=problem.n_eq,
    fixed_elements=problem.fixed_elements,
    sparsity_patterns=sparsity_patterns,
)
mpc = problem.with_solver(solver)


# ======================================================================
# True and nominal dynamics, wrapped in `Dynamics`
# ======================================================================

v_w = Variable("w", (nx,))

def true_step(state, action, params):
    return A_d @ state + (B_d @ action[:, None]).squeeze(-1) + params["w"]

def nominal_step(state, action, _params):
    return A_d @ state + (B_d @ action[:, None]).squeeze(-1)

plant = Dynamics(
    nx=nx,nu=nu,
    true_fun=true_step,
    nominal_fun=nominal_step,
    true_params_spec=(v_w,),
    nominal_params_spec=(),
)


# ======================================================================
# ClosedLoopSimulator wiring
# ======================================================================

def init(inputs, n_steps):
    prepared = mpc.prepare({"cost_state": inputs["cost_state"]})
    return {"x": inputs["x0"], "prepared": prepared}

def step(carry, k):
    sol = mpc.solve_with_prepared(fast_v={"x0": carry["x"]}, warmstart=None,
                    prepared_qp=carry["prepared"])
    u = sol["x"]["u0"]
    x_next = plant.step(
        carry["x"], u,
        true_params={"w": w_true},
        nominal_params={},
    )
    new_carry = {**carry, "x": x_next}
    log = {"x": carry["x"], "u": u}
    return new_carry, log

def finalize(final_carry, logs):
    traj = jnp.concatenate([logs["x"], final_carry["x"][None]], axis=0)
    return {"traj": traj, "u": logs["u"]}


# ======================================================================
# 1. Verbose rollout for illustration
# ======================================================================

rollout_logger = RunLogger()

def _print(k, log):
    if int(k) % 10 == 0:
        x, u = np.asarray(log["x"]), float(log["u"][0])
        rollout_logger.log(k=int(k), x0=x[0], x1=x[1], u0=u)

sim_verbose = ClosedLoop(
    init=init, step=step, n_steps=N_sim, finalize=finalize,
    on_step=_print,
)

print(f"\nClosed loop (N_sim={N_sim}):")
result = sim_verbose.run({"x0": x0_init, "cost_state": cost_state0})
jax.block_until_ready(jax.tree_util.tree_leaves(result))

traj = result["traj"]
print(f"\n  Initial x = [{traj[0, 0]:.4f}, {traj[0, 1]:.4f}]")
print(f"  Final   x = [{traj[-1, 0]:.4f}, {traj[-1, 1]:.4f}]")

archive = TrajectoryStorage()
archive["x_traj"] = traj
archive["u_traj"] = result["u"]
archive["cost_state"] = cost_state0
archive["horizon"] = N
archive.save(".results/sparse_simulator_run")
print(f"  --> Saved verbose rollout data to '.results/sparse_simulator_run'")


# ======================================================================
# 2. JVP of the closed-loop trajectory w.r.t. cost_state
# ======================================================================

sim_silent = ClosedLoop(
    init=init, step=step, n_steps=N_sim, finalize=finalize,
)

def trajectory_of(cost_state):
    return sim_silent.run({"x0": x0_init, "cost_state": cost_state})["traj"]

trajectory_jit = jax.jit(trajectory_of)

def jvp_wrt_cost_state(cost_state, dcs):
    return jax.jvp(trajectory_of, (cost_state,), (dcs,))

jvp_jit = jax.jit(jvp_wrt_cost_state)

print("\nJVP of trajectory w.r.t. cost_state (vs central finite difference):")

jvp_logger = RunLogger()

for i, name in enumerate(["cost_state[0]", "cost_state[1]"]):
    dcs = jnp.zeros(nx).at[i].set(1.0)

    traj_val, dtraj = jvp_jit(cost_state0, dcs)
    jax.block_until_ready((traj_val, dtraj))

    eps     = 1e-4
    traj_p  = trajectory_jit(cost_state0 + eps * dcs)
    traj_m  = trajectory_jit(cost_state0 - eps * dcs)
    dtraj_fd = (traj_p - traj_m) / (2 * eps)

    rel_err = float(jnp.linalg.norm(dtraj - dtraj_fd)
                    / (jnp.linalg.norm(dtraj_fd) + 1e-12))
    cos_sim = float(jnp.dot(dtraj.ravel(), dtraj_fd.ravel())
                    / (jnp.linalg.norm(dtraj)
                       * jnp.linalg.norm(dtraj_fd) + 1e-12))

    jvp_logger.log(
        Parameter=name,
        Norm_dtraj=float(jnp.linalg.norm(dtraj)),
        Rel_Err_FD=rel_err,
        Cos_Sim_FD=cos_sim
    )