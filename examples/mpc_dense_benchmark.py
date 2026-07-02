#!/usr/bin/env python3
"""Standalone MPC benchmarking script for solve and derivative (JVP) performance.

This script isolates the MPC problem from the simulator to find computational bottlenecks.
It allows testing the impact of the "prepared" architecture by toggling whether
`cost_state` is treated as a slow, pre-compiled variable or a fast, dynamically
updated variable.

Usage:
    python mpc_benchmark.py                # Runs with use_prepared = True
    python mpc_benchmark.py --no-prepared  # Runs with use_prepared = False
"""

import time
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision for stable QP solves
jax.config.update("jax_enable_x64", True)

from bpmpc.mpc import Cost, Constraint, MPCProblem
from bpmpc.mpc.helpers import lti_dynamics, box_bounds
from bpmpc.variable import Variable
from jaxsparrow import setup_dense_solver


def run_benchmark(use_prepared: bool, n_warmup: int = 5, n_evals: int = 100):
    print("=" * 60)
    print(f"🚀 Running MPC Benchmark (use_prepared={use_prepared})")
    print("=" * 60)

    # ======================================================================
    # Problem Data
    # ======================================================================
    nx, nu, N   = 2, 1, 50
    cost_input  = 0.1

    A_d = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B_d = jnp.array([[0.0], [1.0]])

    x_min = jnp.array([-5.0, -5.0]); x_max = jnp.array([5.0, 5.0])
    u_min = jnp.array([-0.5]);       u_max = jnp.array([0.5])

    v_x0         = Variable("x0",         (nx,))
    v_cost_state = Variable("cost_state", (nx,))

    # ======================================================================
    # MPC Construction
    # ======================================================================
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

    # Determine slow_vars based on the user toggle
    slow_vars = ["cost_state"] if use_prepared else []

    problem = MPCProblem(
        costs=[mpc_cost],
        constraints=[mpc_dynamics, mpc_bounds],
        slow_vars=slow_vars,
        outputs={"u0": slice(N * nx, N * nx + nu)},
        mode="dense",
    )

    solver = setup_dense_solver(
        n_var=problem.n_dec,
        n_ineq=problem.n_ineq,
        n_eq=problem.n_eq,
        fixed_elements=problem.fixed_elements,
    )
    mpc = problem.with_solver(solver)

    # ======================================================================
    # Function Targets
    # ======================================================================
    x0_init     = jnp.array([-3.0, -1.0])
    cost_state0 = jnp.array([1.0, 1.0])

    if use_prepared:
        # Pre-compute the slow variables once
        prepared_qp = mpc.prepare({"cost_state": cost_state0})

        # 1. Fast solve target (x0 changes, cost_state is already folded into prepared_qp)
        def fast_solve_fn(x0):
            return mpc.solve_with_prepared(prepared_qp=prepared_qp, fast_v={"x0": x0})["x"]["u0"]

        # 2. Derivative target (derivatives wrt cost_state must flow THROUGH prepare)
        def full_solve_for_jvp(cs):
            prep = mpc.prepare({"cost_state": cs})
            return mpc.solve_with_prepared(prepared_qp=prep, fast_v={"x0": x0_init})["x"]["u0"]
    else:
        # 1. Fast solve target (both x0 and cost_state are evaluated on every call)
        def fast_solve_fn(x0):
            return mpc.solve(fast_v={"x0": x0, "cost_state": cost_state0})["x"]["u0"]

        # 2. Derivative target
        def full_solve_for_jvp(cs):
            return mpc.solve(fast_v={"x0": x0_init, "cost_state": cs})["x"]["u0"]

    # JIT the targets
    fast_solve_jit = jax.jit(fast_solve_fn)

    dcs = jnp.array([1.0, 0.0]) # Direction for JVP
    def jvp_fn(cs, direction):
        return jax.jvp(full_solve_for_jvp, (cs,), (direction,))
    
    jvp_jit = jax.jit(jvp_fn)

    # ======================================================================
    # Benchmarking: Forward Solve
    # ======================================================================

    solver.timings.reset()

    print("\n[1/2] Benchmarking Forward Solve...")
    t0 = time.perf_counter()
    _ = fast_solve_jit(x0_init).block_until_ready()
    t1 = time.perf_counter()
    print(f"  Compile time: {t1 - t0:.3f} s")

    for _ in range(n_warmup):
        _ = fast_solve_jit(x0_init).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(n_evals):
        _ = fast_solve_jit(x0_init).block_until_ready()
    t1 = time.perf_counter()
    
    print(f"  Average execution time: {(t1 - t0) / n_evals * 1000:.3f} ms / solve")
    print(solver.timings.summary())

    # ======================================================================
    # Benchmarking: Derivative (JVP)
    # ======================================================================

    solver.timings.reset()

    print("\n[2/2] Benchmarking JVP (Derivative w.r.t cost_state)...")
    t0 = time.perf_counter()
    _, _ = jvp_jit(cost_state0, dcs)
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    print(f"  Compile time: {t1 - t0:.3f} s")

    for _ in range(n_warmup):
        _, _ = jvp_jit(cost_state0, dcs)
        jax.block_until_ready(_)

    t0 = time.perf_counter()
    for _ in range(n_evals):
        _, _ = jvp_jit(cost_state0, dcs)
        jax.block_until_ready(_)
    t1 = time.perf_counter()
    
    print(f"  Average execution time: {(t1 - t0) / n_evals * 1000:.3f} ms / jvp")
    print(solver.timings.summary())
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPC Benchmarking Utility")
    parser.add_argument('--no-prepared', dest='use_prepared', action='store_false',
                        help='Disable the prepare step (treats cost_state as a fast variable)')
    parser.add_argument('--evals', type=int, default=100, 
                        help='Number of iterations for averaging (default: 100)')
    parser.set_defaults(use_prepared=True)
    
    args = parser.parse_args()
    
    run_benchmark(use_prepared=args.use_prepared, n_evals=args.evals)