**Tags:** #class

## Overview

The `closed_loop` module provides a generalized, JAX-accelerated simulation harness. While primarily designed to evaluate MPC controllers, its architecture is completely controller-agnostic. It wraps the highly efficient `jax.lax.scan` operation to run fast, compiled simulations over a given time horizon. The simulation returns a JAX-registered `SimulationResult` PyTree, making the pipeline natively compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

## Purpose

Writing custom `jax.lax.scan` loops for every new problem can be error-prone and tedious. The `ClosedLoop` class abstracts away this boilerplate by enforcing a strict three-phase structure:

* **Init:** A pure Python setup phase that runs before compilation to process inputs, strictly separating them into a time-varying `state` (the loop carry) and read-only static `constants`.
* **Step:** The functionally pure inner loop that executes at every time step. It receives the current state, `x_user` (the time step integer or timeline slice), and static constants, returning the updated state alongside the data to be logged.
* **Finalize:** A pure Python teardown phase that processes the terminal state, accumulated trajectory logs, and constants to compute a scalar objective (for automatic differentiation) and aggregated metrics.

## Key Parameters & Methods

* **`__init__(init, step, n_steps, finalize, on_step=None)`**: Configures the loop with the core phase functions, the total number of steps, and an optional `on_step` Python callback invoked at every step via `jax.debug.callback`.
* **`run(inputs, timeline)`**: Triggers the simulation. It seamlessly handles both static horizons and time-varying exogenous inputs by slicing a provided `timeline` PyTree at each step and passing it to the user-defined `step` function. It returns a `SimulationResult` object containing the objective, trajectories, final state, and metrics.

## Dependencies

This class is strictly uncoupled from the rest of the framework. It acts as a pure utility leaf node and has no internal dependencies on other core classes.