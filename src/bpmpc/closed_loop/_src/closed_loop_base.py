"""Closed-loop simulator for MPC (or any other controller).

A simulation is structurally divided into three distinct phases:

* **Init** — Executes once in Python before the loop. It processes the initial 
                 user inputs and strictly separates them into a time-varying `state` 
                 (the loop carry) and read-only static `constants`.
* **Step** — Executes at every time step. Must be functionally pure as it is 
                 compiled into a ``jax.lax.scan``. It receives the current state 
                 and static constants, returning the updated state and the data to 
                 be logged for that step.
* **Finalize** — Executes once in Python after the loop. It processes the terminal 
                 state, the accumulated trajectory logs, and the constants to compute 
                 a scalar objective (for AD) and aggregated metrics.

The ``step`` function receives either the current step index ``k`` or a slice of a 
user-provided ``timeline`` PyTree. This enables time-varying dynamics, dynamic 
reference trajectories, or exogenous signals.

The simulator is strictly uncoupled from :class:`MPCProblem`; it works for any 
controller architecture. The entire simulation returns a JAX-registered 
:class:`SimulationResult` PyTree, making the pipeline natively compatible with 
``jax.jit``, ``jax.vmap``, and ``jax.grad``.

Example
-------
::

    def init(inputs, n_steps):
        # Prepare the MPC problem once before the loop begins
        prepared = mpc.prepare({"Q_diag": inputs["Q_diag"]})
        
        # Split into time-varying state and read-only constants
        state = {"x": inputs["x0"], "warmstart": None}
        constants = {"prepared": prepared}
        return state, constants

    def step(state, k, constants):
        # Solve the MPC step using the prepared QP from constants
        sol  = mpc.solve({"x0": state["x"]},
                         warmstart=state["warmstart"],
                         prepared_qp=constants["prepared"])
        u0   = sol["x"]["u0"]
        
        # Advance the dynamics
        x_next = A_d @ state["x"] + B_d @ u0
        
        new_state = {"x": x_next, "warmstart": sol["status"]}
        log = {"x": state["x"], "u": u0, "cost": sol["cost"]}
        
        return new_state, log

    def finalize(final_state, logs, constants):
        # Compute differentiable objective and auxiliary metrics
        total_cost = jnp.sum(logs["cost"])
        return total_cost, {"final_x": final_state["x"]}

    sim = ClosedLoop(init=init, step=step, n_steps=50, finalize=finalize)
    results = sim.run({"x0": jnp.array([1.0, 0.0]), "Q_diag": jnp.ones(2)})
    
    print(f"Objective: {results.objective}")
"""

from __future__ import annotations
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from bpmpc.closed_loop._src.simulation_result import SimulationResult

class ClosedLoop:
    """A flexible, JAX-compiled simulation loop for dynamical systems.

    This class encapsulates a `jax.lax.scan` loop, providing hooks to 
    initialize state, execute control steps, and finalize data. It cleanly 
    separates time-varying loop state from static constants to avoid JAX 
    recompilation issues, and returns a structured, differentiable PyTree.

    Parameters
    ----------
    init : Callable[[Any, int], Tuple[Any, Any]]
        Function to initialize the simulation loop.
        Signature: ``init(inputs, n_steps) -> (initial_state, constants)``
        - ``initial_state``: The time-varying variables passed through the loop (the "carry").
        - ``constants``: Static variables (e.g., MPC prep objects) passed to step read-only.
    step : Callable[[Any, Any, Any], Tuple[Any, Any]]
        Function executed at each time step. Must be functionally pure.
        Signature: ``step(state, x_user, constants) -> (new_state, log)``. 
        - ``x_user``: Either the integer time step ``k``, or a time-slice 
          from the ``timeline`` provided to ``run``.
    n_steps : int
        The total number of time steps to simulate.
    finalize : Callable[[Any, Any, Any], Tuple[Any, Any]]
        Function to process the simulation results after the loop finishes.
        Signature: ``finalize(final_state, stacked_logs, constants) -> (objective, metrics)``.
        - ``objective``: A scalar value used for gradient-based optimization.
        - ``metrics``: Any additional aggregated data or dictionary of costs.
    on_step : Callable[[int, Any], None], optional
        A Python callback invoked at every step via ``jax.debug.callback``.
        Useful for printing debug information or using terminal loggers.
        Signature: ``on_step(k, log) -> None``.
    """

    def __init__(
        self,
        init:     Callable[[Any, int], Tuple[Any, Any]],
        step:     Callable[[Any, Any, Any], Tuple[Any, Any]],
        n_steps:  int,
        finalize: Callable[[Any, Any, Any], Tuple[Any, Any]],
        on_step:  Optional[Callable[[int, Any], None]] = None,
    ) -> None:
        self._init     = init
        self._step     = step
        self._finalize = finalize
        self._n_steps  = n_steps
        self._on_step  = on_step

    def run(self, inputs: Any, timeline: Optional[Any] = None) -> SimulationResult:
        """Executes the closed-loop simulation.

        Parameters
        ----------
        inputs : Any
            The initial input data required by the ``init`` function to 
            construct the initial state and constants.
        timeline : Any, optional
            A PyTree of arrays representing time-varying parameters (e.g., 
            reference trajectories, disturbances). Every leaf in this PyTree 
            must have a leading dimension equal to ``n_steps``. If provided, 
            the ``step`` function receives the slice ``timeline[k]`` at each 
            step. If ``None``, the ``step`` function receives the integer index ``k``.

        Returns
        -------
        SimulationResult
            A structured PyTree containing the scalar objective, the stacked 
            trajectories, the terminal state, and aggregated metrics.
        """
        ks = jnp.arange(self._n_steps)
        
        # Determine what x_user is passed to the step function based on the timeline
        if timeline is not None:
            # Assuming _validate_timeline is still available in your helpers
            _validate_timeline(timeline, self._n_steps)
            xs = (ks, timeline)
            user_arg = lambda pair: pair[1]  # Pass the timeline slice
        else:
            xs = (ks, None)
            user_arg = lambda pair: pair[0]  # Pass the step index k

        # 1. Initialize varying states and static constants separately
        state0, constants = self._init(inputs, self._n_steps)

        # 2. Define the functionally pure loop body for jax.lax.scan
        def body(state: Any, pair: Tuple[int, Any]) -> Tuple[Any, Any]:
            k = pair[0]
            x_user = user_arg(pair)
            
            # Execute one simulation step
            new_state, log = self._step(state, x_user, constants)
            
            # Asynchronous callback to the host Python environment (for logging)
            if self._on_step is not None:
                jax.debug.callback(self._on_step, k, log)
                
            return new_state, log

        # 3. Execute the compiled loop
        final_state, logs = jax.lax.scan(body, state0, xs)

        # 4. Finalize the outputs (calculating differentiable objective and metrics)
        objective, metrics = self._finalize(final_state, logs, constants)

        # 5. Pack everything into the standard return object
        return SimulationResult(
            objective=objective,
            trajectory=logs,
            final_state=final_state,
            metrics=metrics
        )


# ======================================================================
# Helpers
# ======================================================================

def _validate_timeline(timeline: Any, n_steps: int) -> None:
    """Validates that every leaf in a timeline PyTree spans the correct horizon.

    Parameters
    ----------
    timeline : Any
        The user-provided PyTree of time-varying arrays.
    n_steps : int
        The expected length of the leading dimension.

    Raises
    ------
    ValueError
        If the timeline is empty, or if any leaf does not have ``n_steps`` 
        as its leading dimension.
    """
    leaves = jax.tree_util.tree_leaves(timeline)
    if not leaves:
        raise ValueError("timeline pytree is empty.")
    for leaf in leaves:
        shape = jnp.shape(leaf)
        if not shape or shape[0] != n_steps:
            raise ValueError(
                f"Timeline leaves must have leading dimension {n_steps}, "
                f"got shape {shape}."
            )