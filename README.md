# bpmpc: Differentiable Model Predictive Control in JAX

`bpmpc_jax` is a high-performance, fully differentiable Model Predictive Control (MPC) framework built natively in JAX. It provides a modular, symbolic API to design parametric optimization problems, compiles them into blazing-fast XLA executables, and allows you to differentiate directly through your closed-loop control trajectories.

Whether you are tuning cost weights via gradient descent, integrating learning-based dynamics, or simply seeking a lightning-fast MPC solver, `bpmpc_jax` is built to be mathematically rigorous and highly optimized.

---

## 🚀 Core Features

* **Symbolic Problem Definition:** Declare parameters as `Variable`s and separate your problem into mathematical terms. No more manual, error-prone matrix stacking.
* **Fully Differentiable:** Native support for `jax.jit`, `jax.jvp`, and `jax.grad`. Differentiate closed-loop trajectories with respect to any parameter (costs, bounds, nominal dynamics).
* **Two-Tiered Compilation:** Strict separation of "slow" parameters (compiled once per episode) and "fast" parameters (patched at every time step) to maximize control frequency.
* **Straight-Through Dynamics:** Safely decouple your true forward simulation (e.g., noisy environments) from your backward AD surrogate models.

---

## 🛠️ Quickstart Guide

### 1. Define the Problem (Costs & Constraints)

Instead of manually building block-diagonal matrices, use the built-in helper factories. They handle time-tiling and matrix stacking automatically.

```python
import jax
import jax.numpy as jnp
from bpmpc_jax.variable import Variable
from bpmpc_jax.mpc.helpers import lti_dynamics, box_bounds
from bpmpc_jax.mpc import Cost, MPCProblem
from jaxsparrow import setup_dense_solver

# System dimensions
nx, nu, N = 2, 1, 50

# System dynamics (Simple double integrator)
A_d = jnp.array([[1.0, 0.1], [0.0, 1.0]])
B_d = jnp.array([[0.0], [0.1]])

# 1. Define symbolic parameters
v_x0 = Variable("x0", (nx,))
v_cost_state = Variable("cost_state", (nx,))

# 2. Build dynamics and bounds constraints
mpc_dynamics = lti_dynamics(A=A_d, B=B_d, x0=v_x0, horizon=N)

# Easily convert hard bounds to soft constraints by chaining .slack()
mpc_bounds = box_bounds(
    x_min=jnp.array([-5.0, -5.0]), x_max=jnp.array([5.0, 5.0]),
    u_min=jnp.array([-0.5]),       u_max=jnp.array([0.5]),
    horizon=N
).slack(w_quad=1e4, w_lin=1e2) 

# 3. Define a custom tracking cost
def q_mat(v):
    Q = jnp.diag(v["cost_state"])
    R = 0.1 * jnp.eye(nu)
    return jnp.block([
        [jnp.kron(jnp.eye(N), Q), jnp.zeros((N * nx, N * nu))],
        [jnp.zeros((N * nu, N * nx)), jnp.kron(jnp.eye(N), R)],
    ])

mpc_cost = Cost(q_mat=q_mat, v_in_q_mat={"cost_state": v_cost_state})
```

### 2. Compile the Solver

Bundle your definitions into an `MPCProblem`. Crucially, declare which variables update slowly to optimize the underlying JAX compilation.

```python
# Build the problem and separate slow vs. fast variables
problem = MPCProblem(
    costs=[mpc_cost],
    constraints=[mpc_dynamics, mpc_bounds],
    slow_vars=["cost_state"],
    outputs={"u0": slice(N * nx, N * nx + nu)}, # Automatically extract first control action
    mode="dense"
)

# Attach a numerical backend (e.g., jaxsparrow)
solver_backend = setup_dense_solver(
    n_var=problem.n_dec,
    n_ineq=problem.n_ineq,
    n_eq=problem.n_eq,
    fixed_elements=problem.fixed_elements,
)
mpc = problem.with_solver(solver_backend)
```

### 3. Solve and Differentiate

The `MPCSolver` exposes a clean, partitioned API. `prepare` bakes slow parameters into a base QP, while `solve_with_prepared` rapidly injects fast variables (like the current state).

```python
# Step A: Prepare slow variables (e.g., once per episode)
cost_weights = jnp.array([1.0, 1.0])
prepared_qp = mpc.prepare({"cost_state": cost_weights})

# Step B: Solve for fast variables (e.g., at every timestep)
current_state = jnp.array([-3.0, -1.0])
solution = mpc.solve_with_prepared(
    prepared_qp=prepared_qp,
    fast_v={"x0": current_state}, 
    warmstart=None
)
optimal_action = solution["x"]["u0"]

# Step C: Fully differentiable through the solver!
def solve_action(cost_weights_dynamic):
    prep = mpc.prepare({"cost_state": cost_weights_dynamic})
    sol = mpc.solve_with_prepared(prep, {"x0": current_state}, None)
    return sol["x"]["u0"]

# Compute the Jacobian of the control action w.r.t the cost weights
action_jacobian = jax.jacfwd(solve_action)(cost_weights)
```

### 4. Closed-Loop Simulation & Dynamics

Simulate your controller robustly using the `ClosedLoop` module. It compiles the entire simulation loop using `jax.lax.scan` for massive performance gains.

```python
from bpmpc_jax.dynamics import Dynamics
from bpmpc_jax.closed_loop import ClosedLoop

# Decouple true noisy environment from smooth AD surrogate
def true_step(state, action, params):
    return A_d @ state + B_d @ action + params["drift"]

def nominal_step(state, action, params):
    return A_d @ state + B_d @ action

plant = Dynamics(true_fun=true_step, nominal_fun=nominal_step)

# Define loop mechanics
def init(inputs, n_steps):
    prep = mpc.prepare({"cost_state": inputs["cost_weights"]})
    return {"x": inputs["x0"], "prepared": prep}

def step(carry, k):
    sol = mpc.solve_with_prepared(carry["prepared"], {"x0": carry["x"]}, None)
    u = sol["x"]["u0"]
    x_next = plant.step(carry["x"], u, {"drift": 0.02}, {})
    
    return {**carry, "x": x_next}, {"x": carry["x"], "u": u}

# Run compiled simulation
sim = ClosedLoop(init=init, step=step, n_steps=80)
result = sim.run({"x0": current_state, "cost_weights": cost_weights})
```

## 🌌 Scaling Up: Sparse Mode

For problems with long prediction horizons or large state dimensions, dense matrices are heavily populated with zeros, wasting memory and computing power. `bpmpc_jax` provides a highly optimized sparse backend powered by XLA-fused `jax.experimental.sparse.BCOO` matrices.

The best part? You don't have to manually define the structural sparsity pattern. The framework uses **random probing** under the hood—evaluating your custom cost and constraint functions with dummy noise at compilation time to automatically discover the exact memory footprint of your non-zero elements.

Switching from dense to sparse mode requires only two simple changes: setting `mode="sparse"` and attaching the sparse solver backend.

```python
from bpmpc_jax.mpc import MPCProblem
from jaxsparrow import setup_sparse_solver

# 1. Set mode="sparse" when defining the problem
sparse_problem = MPCProblem(
    costs=[mpc_cost],
    constraints=[mpc_dynamics, mpc_bounds],
    slow_vars=["cost_state"],
    outputs={"u0": slice(N * nx, N * nx + nu)},
    mode="sparse"  # <-- The framework auto-discovers structural sparsity here
)

# 2. Attach the sparse numerical backend
sparse_solver_backend = setup_sparse_solver(
    n_var=sparse_problem.n_dec,
    n_ineq=sparse_problem.n_ineq,
    n_eq=sparse_problem.n_eq,
    fixed_elements=sparse_problem.fixed_elements,
)
sparse_mpc = sparse_problem.with_solver(sparse_solver_backend)
```

Once compiled, the `MPCSolver` API (`prepare` and `solve_with_prepared`) remains **exactly the same**. The assembler will now perform extremely fast, 1D data array updates without ever rebuilding the underlying matrix graph connectivity.
