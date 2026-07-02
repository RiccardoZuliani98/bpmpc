import time
from typing import Any, Callable, Iterable, Optional, List, Tuple, Dict

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from bpmpc.closed_loop import ClosedLoop, RunLogger

# ============================================================================
# 1. Tuning Helper
# ============================================================================

def closed_loop_tune(
    loss_fn: Callable,
    initial_params: Any,
    optimizer: Optional[optax.GradientTransformation] = None,
    n_iters: int = 50,
    dataloader: Optional[Iterable[Any]] = None,
    batch_size: int = 1,
    has_aux: bool = False,
) -> Tuple[Any, Optional[List[Any]]]:
    """Optimizes arbitrary parameters using a differentiable loss function.
    
    Parameters
    ----------
    loss_fn : Callable
        The function computing the loss. 
        - If `dataloader` is None: `loss_fn(params) -> loss`
        - If `dataloader` is provided: `loss_fn(params, item) -> loss`
        - If `has_aux=True`, it must return `(loss, aux_data)`.
    initial_params : Any
        The starting parameters (can be a dict, dataclass, or JAX array).
    optimizer : optax.GradientTransformation, optional
        The Optax optimizer to use. Defaults to Adam with gradient clipping.
    n_iters : int
        Number of optimization steps (parameter updates).
    dataloader : Iterable[Any], optional
        An iterator or generator yielding individual data items for stochastic updates.
    batch_size : int, optional
        Number of items to accumulate gradients over before making an optimizer update.
    has_aux : bool, optional
        If True, expects `loss_fn` to return a tuple `(loss, aux_data)`. 
        Collects `aux_data` over all iterations.
        
    Returns
    -------
    params : Any
        The final optimized parameters.
    aux_history : List or None
        The collected auxiliary data if `has_aux=True`, otherwise `None`.
    """
    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=1e-2)
        )

    opt_state = optimizer.init(initial_params)
    params = initial_params
    aux_history = [] if has_aux else None
    
    logger = RunLogger(formats={
        "Epoch": "{:02d}", 
        "Step": "{:02d}", 
        "TaskCost": "{:.2f}", 
        "Time": "{:.3f}s"
    })

    # ==========================================
    # BRANCH 1: DETERMINISTIC MODE (No Batching)
    # ==========================================
    if dataloader is None:
        backward = jax.jit(jax.value_and_grad(loss_fn, has_aux=has_aux))
        
        def train_step_single(p, state):
            if has_aux:
                (loss_val, aux), grads = backward(p)
            else:
                loss_val, grads = backward(p)
                aux = None
                
            updates, new_state = optimizer.update(grads, state, p)
            new_p = optax.apply_updates(p, updates)
            return new_p, new_state, loss_val, aux

        print("Starting differentiable tuning (Deterministic mode)...")
        for epoch in range(n_iters):
            t0 = time.time()
            params, opt_state, loss_val, aux = train_step_single(params, opt_state)
            
            if has_aux:
                aux_history.append(aux)  # type: ignore
                
            # NEW SYNTAX: Pass raw values; the logger handles .item() and formatting
            logger.log(Epoch=epoch, TaskCost=loss_val, Time=time.time() - t0)

    # ==========================================
    # BRANCH 2: STOCHASTIC MODE (Gradient Accumulation)
    # ==========================================
    else:
        backward = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=has_aux))
        print(f"Starting differentiable tuning (Gradient Accumulation, batch={batch_size})...")
        batch_iter = iter(dataloader)
        
        for step in range(n_iters):
            t0 = time.time()
            accumulated_grads = None
            total_loss = 0.0
            step_aux = []
            
            for _ in range(batch_size):
                try:
                    item = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    item = next(batch_iter)
                
                if has_aux:
                    (loss_val, aux), grads = backward(params, item)
                    step_aux.append(aux)
                else:
                    loss_val, grads = backward(params, item)
                    
                total_loss += loss_val
                
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jtu.tree_map(lambda x, y: x + y, accumulated_grads, grads)
            
            if has_aux:
                aux_history.append(step_aux)  # type: ignore
            
            avg_loss = total_loss / batch_size
            avg_grads = jtu.tree_map(lambda x: x / batch_size, accumulated_grads)
            
            updates, opt_state = optimizer.update(avg_grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # NEW SYNTAX: Pass raw values; the logger handles .item() and formatting
            logger.log(Step=step, TaskCost=avg_loss, Time=time.time() - t0)

    print("\nTuning complete.")
    return params, aux_history


# ============================================================================
# 2. Simulator Builder
# ============================================================================

def build_closed_loop_simulator(
    mpc: Any, 
    plant: Any, 
    trajectory_cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Any], 
    horizon_sim: int, 
    horizon_mpc: int, 
    nx: int, 
    nu: int,
    runtime_keys: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None
) -> ClosedLoop:
    """Constructs a JAX-compatible ClosedLoop simulation object.

    Parameters
    ----------
    mpc : object
        The Model Predictive Control configuration object.
    plant : object
        The true system dynamics model.
    trajectory_cost_fn : Callable
        Function returning a scalar cost objective from `(xs, us)`.
    horizon_sim : int
        The total number of timesteps to simulate.
    horizon_mpc : int
        The prediction horizon length used by the MPC internally.
    nx : int
        The dimension of the state vector.
    nu : int
        The dimension of the control input vector.
    runtime_keys: List[str], optional
        Parameter names in `inputs` treated as time-varying sequences 
        of length `horizon_sim`. These are sliced at each timestep `k`.
    options : dict, optional
        Additional configuration options (e.g., `linearization_mode`).

    Returns
    -------
    ClosedLoop
        A compiled simulation factory ready to be executed via `.run(inputs)`.
    """
    if options is None:
        options = {}
        
    warmstart_first_mpc = options.get("warmstart_first_mpc", False)
    lin_mode = options.get("linearization_mode", "none")

    valid_modes = ["trajectory_shifted", "trajectory", "current_state", "none"]
    if lin_mode not in valid_modes:
        raise ValueError(f"Unknown linearization_mode: '{lin_mode}'. Valid options: {valid_modes}")

    mpc_keys = set(mpc.all_vars.keys())
    plant_keys = set(plant.true_params_spec.keys())
    rt_keys_set = set(runtime_keys) if runtime_keys is not None else set()

    def init(inputs: Dict[str, Any], n_steps: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        x0 = inputs["x0"]
        
        # Split static vs runtime sequences
        mpc_static    = {k: inputs[k] for k in mpc_keys if k in inputs and k not in rt_keys_set}
        mpc_runtime   = {k: inputs[k] for k in mpc_keys if k in inputs and k in rt_keys_set}
        plant_static  = {k: inputs[k] for k in plant_keys if k in inputs and k not in rt_keys_set}
        plant_runtime = {k: inputs[k] for k in plant_keys if k in inputs and k in rt_keys_set}

        # Evaluate runtime sequences at k=0 to initialize the first problem
        mpc_params_k0 = {**mpc_static}
        for key, seq in mpc_runtime.items():
            mpc_params_k0[key] = seq[0]

        prep = mpc.prepare(mpc_params_k0)
        solve_args = {"x0": x0, **mpc_params_k0}
        
        initial_state = {"x": x0}
        
        # NEW SYNTAX: Fixed variables that do not change shape or type inside the loop
        constants = {
            "prepared": prep,
            "mpc_static": mpc_static,
            "mpc_runtime": mpc_runtime,
            "plant_static": plant_static,
            "plant_runtime": plant_runtime
        }
        
        if lin_mode != "none":
            x_seed = jnp.tile(x0, (horizon_mpc, 1))
            u_seed = jnp.zeros((horizon_mpc, nu))
            solve_args["x_nom"] = x_seed
            solve_args["u_nom"] = u_seed
        
        if warmstart_first_mpc:
            sol0 = mpc.solve_with_prepared(prep, solve_args, warmstart=None)
            if lin_mode != "none":
                initial_state["x_nom"] = sol0["x"]["x"].reshape((horizon_mpc, nx))
                initial_state["u_nom"] = sol0["x"]["u"].reshape((horizon_mpc, nu))
        else:
            if lin_mode != "none":
                initial_state["x_nom"] = x_seed # type: ignore
                initial_state["u_nom"] = u_seed # type: ignore
            
        return initial_state, constants

    def step(state: Dict[str, Any], k: int, constants: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Assemble current MPC parameters by dynamically slicing the runtime sequences at k
        current_mpc_params = {**constants["mpc_static"]}
        for key, seq in constants["mpc_runtime"].items():
            current_mpc_params[key] = seq[k]
            
        solve_args = {"x0": state["x"], **current_mpc_params}
        
        if lin_mode != "none":
            solve_args["x_nom"] = state["x_nom"]
            solve_args["u_nom"] = state["u_nom"]

        sol = mpc.solve_with_prepared(constants["prepared"], solve_args, warmstart=None)
        u = sol["x"]["u0"]
        
        # Assemble current Plant parameters
        current_plant_params = {**constants["plant_static"]}
        for key, seq in constants["plant_runtime"].items():
            current_plant_params[key] = seq[k]
        
        x_next = plant.step(state["x"], u, true_params=current_plant_params)
        
        # Clean carry update (constants are no longer dragged through)
        new_state = {"x": x_next}
        
        if lin_mode != "none":
            x_sol = sol["x"]["x"].reshape((horizon_mpc, nx))
            u_sol = sol["x"]["u"].reshape((horizon_mpc, nu))
            
            if lin_mode == "trajectory_shifted":
                new_state["u_nom"] = jnp.concatenate([u_sol[1:], u_sol[-1:]], axis=0)
                new_state["x_nom"] = x_sol.at[0].set(x_next)
            elif lin_mode == "trajectory":
                new_state["u_nom"] = u_sol
                new_state["x_nom"] = x_sol.at[0].set(x_next)
            elif lin_mode == "current_state":
                u_next_scalar = u_sol[1] if horizon_mpc > 1 else u_sol[0]
                new_state["u_nom"] = jnp.tile(u_next_scalar, (horizon_mpc, 1))
                new_state["x_nom"] = jnp.tile(x_next, (horizon_mpc, 1))

        # Log dictionary: this is what becomes results.trajectory["x"] / ["u"]
        step_log = {"x": state["x"], "u": u}
        return new_state, step_log

    def finalize(final_state: Dict[str, Any], logs: Dict[str, Any], constants: Dict[str, Any]) -> Tuple[Any, Any]:
        # NEW SYNTAX: return objective, metrics 
        # (xs and us are natively stored in SimulationResult.trajectory, no need to duplicate)
        xs, us = logs["x"], logs["u"]
        objective = trajectory_cost_fn(xs, us)
        return objective, None

    return ClosedLoop(init=init, step=step, n_steps=horizon_sim, finalize=finalize)