"""
Differentiable MPC: tuning cost weights via gradient descent
============================================================

This example shows how to use ``bpmpc`` to tune the internal cost
weights of a Model Predictive Controller end-to-end using Stochastic 
Gradient Descent (SGD) across batches of initial states, model 
uncertainties, and process noise.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import time

jax.config.update("jax_enable_x64", True)

from bpmpc.variable import Variable
from bpmpc.mpc import MPCProblem, Cost
from bpmpc.mpc.helpers import (
    build_state_tracking, 
    nonlinear_dynamics, 
    box_bounds
)
from bpmpc.closed_loop.helpers import (
    build_closed_loop_simulator, 
    quadratic_cost_and_penalty,
    dare_init_theta,
    closed_loop_tune
)
from bpmpc.env import CartPendulum
from jaxsparrow import setup_sparse_solver


# ============================================================================
# 1. Hyperparameters
# ============================================================================
# --- Plant / closed-loop simulation ---
DT          = 0.015
HORIZON_SIM = 170                                          # rollout length
BATCH_SIZE  = 16                                           # SGD batch size

# --- MPC ---
HORIZON_MPC = 11
NX, NU      = 4, 1
TERM_REG    = 1e-2                                         # regularisation on Q_term
R_REG       = 1e-6                                         # regularisation on R

X_MIN_MPC = jnp.array([-5.0, -5.0, -jnp.inf, -jnp.inf])    # MPC state bounds
X_MAX_MPC = jnp.array([ 5.0,  5.0,  jnp.inf,  jnp.inf])
U_MIN_MPC = jnp.array([-4.0])                              # MPC input bounds
U_MAX_MPC = jnp.array([ 4.0])

Q_RUN = jnp.diag(jnp.array([100.0, 1.0, 100.0, 1.0]))      # fixed running cost

# --- Task cost (defines the outer/meta-loss) ---
X0_NOM  = jnp.array([-3.0, 0.0, 0.0, 0.0])                 # Nominal start state
Q_TASK  = jnp.diag(jnp.array([100.0, 1.0, 100.0, 1.0]))
R_TASK  = jnp.diag(jnp.array([1e-6]))
X_MIN_TASK  = jnp.array([-5.0, -5.0, -jnp.inf, -jnp.inf])
X_MAX_TASK  = jnp.array([ 5.0,  5.0,  jnp.inf,  jnp.inf])
VIOLATION_W = 100.0                                        # penalty weight

NX_TRI = NX * (NX + 1) // 2                                # 10
NP     = NX_TRI + NU                                       # 11

# --- Outer optimiser ---
N_ITER = 200
RHO    = 1e-4
ETA    = 0.51
CLIP   = 1e4


# ============================================================================
# 2. Building the MPC problem
# ============================================================================
def setup_swingup_mpc(plant):
    p_var = Variable("p", shape=(NP,))
    x0_var    = Variable("x0",    shape=(NX,))
    x_nom_var = Variable("x_nom", shape=(HORIZON_MPC, NX))
    u_nom_var = Variable("u_nom", shape=(HORIZON_MPC, NU))

    n_z = HORIZON_MPC * NX + HORIZON_MPC * NU
    tril_idx = jnp.tril_indices(NX)

    def _terminal_from_cq(c_q):
        L = jnp.zeros((NX, NX)).at[tril_idx].set(c_q)
        return L @ L.T + TERM_REG * jnp.eye(NX)

    def _get_P(v):
        c_q = v["p"][:NX_TRI]
        c_r = v["p"][NX_TRI:]
        Q_term = _terminal_from_cq(c_q)
        R_mat  = jnp.diag(c_r**2 + R_REG)

        Q_tiled = jnp.broadcast_to(Q_RUN, (HORIZON_MPC + 1, NX, NX))
        Q_tiled = Q_tiled.at[-1].set(Q_term)
        R_tiled = jnp.broadcast_to(R_mat, (HORIZON_MPC, NU, NU))

        P, _, _ = build_state_tracking(
            Q_tiled, R_tiled,
            jnp.zeros((HORIZON_MPC + 1, NX)),
            jnp.zeros((HORIZON_MPC, NU)),
            jnp.zeros((NX,)),
            HORIZON_MPC,
        )
        return P

    cost = Cost(
        q_mat=_get_P,
        q_vec=lambda v: jnp.zeros(n_z),
        c=lambda v: jnp.array(0.0),
        v_in_q_mat={"p": p_var},
        v_in_q_vec=None,
        v_in_c=None,
    )

    dyn = nonlinear_dynamics(
        dyn=plant,
        x_nominal=x_nom_var,
        u_nominal=u_nom_var,
        horizon=HORIZON_MPC,
    )
    bnds_state = box_bounds(
        x_min=X_MIN_MPC, x_max=X_MAX_MPC,
        n_u = U_MIN_MPC.shape[0],
        horizon=HORIZON_MPC,
    ).add_slack(w_quad=10,w_lin=10)
    
    bnds_input = box_bounds(
        u_min=U_MIN_MPC, u_max=U_MAX_MPC,
        n_x = X_MIN_MPC.shape[0],
        horizon=HORIZON_MPC,
    )

    mpc = MPCProblem(
        costs=[cost],
        constraints=[dyn, bnds_state, bnds_input],
        outputs={
            "u0": slice(HORIZON_MPC * NX,       HORIZON_MPC * NX + NU),
            "x":  slice(0,                      HORIZON_MPC * NX),
            "u":  slice(HORIZON_MPC * NX,       HORIZON_MPC * (NX + NU)),
        },
        mode="sparse",
    )

    solver = setup_sparse_solver(
        n_var=mpc.n_dec,
        n_ineq=mpc.n_ineq,
        n_eq=mpc.n_eq,
        sparsity_patterns=mpc.sparsity_pattern,
        fixed_elements=mpc.fixed_elements,
        options={"diff_mode": "rev"},
    )
    return mpc.with_solver(solver)

# Instantiate the environment and ENABLE noise/uncertainty hooks
env   = CartPendulum(dt=DT)
plant = env.get_dynamics(include_w=True, include_d=True)
mpc   = setup_swingup_mpc(plant)


# ============================================================================
# 3. Closed-loop simulation setup using the Cost Builder
# ============================================================================

swingup_task_cost = quadratic_cost_and_penalty(
    Q=Q_TASK,
    R=R_TASK,
    x_min=X_MIN_TASK,
    x_max=X_MAX_TASK,
    w_viol_lin=VIOLATION_W,
    w_viol_quad=0.0
)

simulate_closed_loop_fn = build_closed_loop_simulator(
    mpc=mpc,
    plant=plant,
    trajectory_cost_fn=swingup_task_cost,
    horizon_sim=HORIZON_SIM,
    horizon_mpc=HORIZON_MPC,
    nx=NX,
    nu=NU,
    options={
        "warmstart_first_mpc": True, 
        "linearization_mode": "trajectory_shifted"
    }
)


# ============================================================================
# 4. Initialisation: DARE for a sane starting theta
# ============================================================================
theta_init = {"p":jnp.concatenate([
    dare_init_theta(
        plant=plant,
        Q=np.array(Q_RUN),
        R=np.array(R_TASK),
        reg=TERM_REG
    ),
    np.array([1e-3])
])}


# ============================================================================
# 5. Dataloader & Stochastic Loss Setup
# ============================================================================
def data_generator(batch_size, seed=42):
    key = jax.random.PRNGKey(seed)
    while True:
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        
        # 1. Sample Initial State: {0} x [-0.3, 0.3] x {0} x [-0.3, 0.3]
        dx_1 = jax.random.uniform(k1, (batch_size,), minval=-0.3, maxval=0.3)
        dx_3 = jax.random.uniform(k2, (batch_size,), minval=-0.3, maxval=0.3)
        zeros_x = jnp.zeros((batch_size,))
        delta_x0 = jnp.stack([zeros_x, dx_1, zeros_x, dx_3], axis=-1)
        x0_batch = X0_NOM + delta_x0

        # 2. Sample Model Uncertainty (d): [-0.05, 0.05]^2
        d_batch = jax.random.uniform(k3, (batch_size, 2), minval=-0.05, maxval=0.05)

        # 3. Sample Process Noise (w): {0} x [-0.01, 0.01] x {0} x [-0.1, 0.1]
        w_1 = jax.random.uniform(k4, (batch_size, HORIZON_SIM), minval=-0.01, maxval=0.01)
        w_3 = jax.random.uniform(k5, (batch_size, HORIZON_SIM), minval=-0.1, maxval=0.1)
        zeros_w = jnp.zeros((batch_size, HORIZON_SIM))
        w_batch = jnp.stack([zeros_w, w_1, zeros_w, w_3], axis=-1)

        yield {"x0": x0_batch, "d": d_batch, "w": w_batch}

# Define the single-condition simulator wrapper
def single_sim_loss(p, x0, d, w):
    res = simulate_closed_loop_fn(
        x0=x0, 
        mpc_params=p, 
        plant_params={"d": d}, 
        runtime_plant_params={"w": w}
    )
    return res["cost"]

# Vectorize the simulator over the batch dimensions (x0, d, w) but NOT the params (p)
batch_sim_loss = jax.vmap(single_sim_loss, in_axes=(None, 0, 0, 0))

# The batched loss function for `closed_loop_tune`
def sgd_loss_fn(p, batch):
    costs = batch_sim_loss(p, batch["x0"], batch["d"], batch["w"])
    return jnp.mean(costs)

optimizer = optax.chain(
    optax.clip_by_global_norm(CLIP),
    optax.sgd(learning_rate=lambda k: RHO * jnp.log(k + 2) / (k + 1) ** ETA)
)


# ============================================================================
# 6. Main and plotting
# ============================================================================

if __name__ == "__main__":
    
    # We define a "worst-case" test scenario for the before/after plots
    # using extreme values from our sampling distributions.
    test_x0 = X0_NOM + jnp.array([0.0, 0.3, 0.0, -0.3])
    test_d  = jnp.array([0.05, 0.05])
    test_w  = jnp.stack([
        jnp.zeros(HORIZON_SIM), 
        jnp.full(HORIZON_SIM, 0.01), 
        jnp.zeros(HORIZON_SIM), 
        jnp.full(HORIZON_SIM, 0.1)
    ], axis=-1)

    def run_test_scenario(p):
        return simulate_closed_loop_fn(
            x0=test_x0, 
            mpc_params=p, 
            plant_params={"d": test_d}, 
            runtime_plant_params={"w": test_w}
        )

    # 1. Roll out with the initial DARE-based parameters.
    print("Rolling out Test Scenario with initial parameters...")
    traj_before = run_test_scenario(theta_init)
    print(f"  Initial test scenario cost: {float(traj_before['cost']):.2f}\n")

    # 2. Tune using Batched SGD
    theta_final = closed_loop_tune(
        loss_fn=sgd_loss_fn,
        initial_params=theta_init,
        optimizer=optimizer,
        n_iters=N_ITER,
        dataloader=data_generator(batch_size=BATCH_SIZE)
    )

    # 3. Roll out with the optimised parameters.
    print("\nRolling out Test Scenario with optimised parameters...")
    traj_after = run_test_scenario(theta_final)
    print(f"  Final test scenario cost: {float(traj_after['cost']):.2f}\n")

    # 4. Compare.
    t = np.arange(HORIZON_SIM) * DT

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.5), sharex=True)

    axes[0].plot(t, traj_before["xs"][:, 0], linestyle="--", color="tab:gray", label="before")
    axes[0].plot(t, traj_after["xs"][:, 0], color="tab:blue", label="after")
    axes[0].axhline(0.0, color="k", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("Cart position [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, traj_before["xs"][:, 2], linestyle="--", color="tab:gray", label="before")
    axes[1].plot(t, traj_after["xs"][:, 2], color="tab:orange", label="after")
    axes[1].axhline( 0.0,    color="g", linewidth=0.6, linestyle=":", label="upright")
    axes[1].axhline(-np.pi,  color="r", linewidth=0.6, linestyle=":", label="hanging")
    axes[1].set_ylabel("Pole angle [rad]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", ncol=2)

    axes[2].plot(t, traj_before["us"][:, 0], linestyle="--", color="tab:gray", label="before")
    axes[2].plot(t, traj_after["us"][:, 0], color="tab:green", label="after")
    axes[2].axhline(float(U_MAX_MPC[0]), color="r", linewidth=0.6, linestyle=":")
    axes[2].axhline(float(U_MIN_MPC[0]), color="r", linewidth=0.6, linestyle=":")
    axes[2].set_ylabel("Control input [N]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle("Robust Closed-loop Swing-up (Tuned via SGD): Before vs After")
    fig.tight_layout()
    plt.show()