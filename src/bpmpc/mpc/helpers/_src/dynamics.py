"""Dynamics equality constraint factories and builders.

Two layers:

1. **Builders** (``build_*``) — pure functions that take concrete
   arrays and return dense matrices.  Useful when the user needs
   custom parametric logic inside a ``Constraint`` lambda.

2. **Factories** (``lti_dynamics``, ``ltv_dynamics``, ``nonlinear_dynamics``) — convenience
   functions that accept ``Array | Variable`` per argument and
   return a ready-to-use :class:`Constraint`.

Decision vector layout::

    z = [x_1; ...; x_N; u_0; ...; u_{N-1}]

where ``x_0`` is a parameter, not a decision variable.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from bpmpc.mpc._src.constraint import Constraint
from bpmpc.mpc.helpers._src.util import ArrayOrVar, resolve, collect_v_in, auto_tile
from bpmpc.dynamics import Dynamics
from bpmpc.mpc._src.partition import Partition


# ======================================================================
# LTI builders
# ======================================================================

def build_lti_lhs(A: Array, B: Array, horizon: int) -> Array:
    """Build the dense equality LHS for time-invariant dynamics.

    Returns ``F`` of shape ``(N*n_x, N*n_x + N*n_u)`` such that
    ``F @ z = rhs`` encodes ``x_{k+1} = A x_k + B u_k + c``.

    The block structure is::

        F_x = -kron(I_N, I_{n_x}) + kron(S, A)
        F_u = kron(I_N, B)
        F   = [F_x | F_u]

    where ``S`` is the sub-diagonal shift matrix.
    """
    n_x, n_u = A.shape[0], B.shape[1]
    N = horizon
    I_N = jnp.eye(N)
    I_x = jnp.eye(n_x)
    S = jnp.eye(N, k=-1)
    Fx = -jnp.kron(I_N, I_x) + jnp.kron(S, A)
    Fu = jnp.kron(I_N, B)
    return jnp.concatenate([Fx, Fu], axis=1)


def build_lti_rhs(
    A: Array, x0: Array, horizon: int,
    c: Optional[Array] = None,
) -> Array:
    """Build the dense equality RHS for time-invariant dynamics.

    Returns ``rhs`` of shape ``(N*n_x,)`` such that
    ``lhs @ z = rhs``.  The first block includes ``-A x_0``
    and every block includes ``-c`` (or zero if ``c`` is None).
    """
    n_x = A.shape[0]
    N = horizon
    x0 = x0.reshape((n_x,))
    c_val = jnp.zeros(n_x) if c is None else c.reshape((n_x,))
    f = jnp.tile(-c_val, N)
    f = f.at[:n_x].add(-(A @ x0))
    return f


# ======================================================================
# LTV builders
# ======================================================================

def build_ltv_lhs(A: Array, B: Array, horizon: int) -> Array:
    """Build the dense equality LHS for time-varying dynamics.

    Parameters
    ----------
    A : ``(N, n_x, n_x)`` — per-step state matrices.
    B : ``(N, n_x, n_u)`` — per-step input matrices.
    horizon : N

    Returns
    -------
    ``(N*n_x, N*n_x + N*n_u)`` dense constraint matrix.
    """
    N = horizon
    n_x = A.shape[1]
    n_u = B.shape[2]
    n_x_total = N * n_x
    n_u_total = N * n_u

    ks = jnp.arange(N)
    i_x = jnp.arange(n_x)
    j_x = jnp.arange(n_x)
    i_u = jnp.arange(n_u)

    fx = -jnp.kron(jnp.eye(N), jnp.eye(n_x))
    row_base = (ks[:, None, None] * n_x) + i_x[None, :, None]
    col_base = (ks[:, None, None] * n_x) + j_x[None, None, :]
    fx = fx.at[row_base[1:], col_base[:-1]].set(A[1:])

    cols_B = (ks[:, None, None] * n_u) + i_u[None, None, :]
    fu = jnp.zeros((n_x_total, n_u_total)).at[row_base, cols_B].set(B)

    return jnp.concatenate([fx, fu], axis=1)


def build_ltv_rhs(
    A: Array, x0: Array, horizon: int,
    c: Optional[Array] = None,
) -> Array:
    """Build the dense equality RHS for time-varying dynamics.

    Parameters
    ----------
    A  : ``(N, n_x, n_x)``
    x0 : ``(n_x,)``
    horizon : N
    c  : ``(N, n_x)`` optional per-step affine term.

    Returns
    -------
    ``(N*n_x,)`` dense RHS vector.
    """
    N = horizon
    n_x = A.shape[1]
    x0 = x0.reshape((n_x,))
    c_val = jnp.zeros((N, n_x)) if c is None else c
    f = (-c_val).reshape((N * n_x,))
    f = f.at[:n_x].add(-(A[0] @ x0))
    return f


# ======================================================================
# LTI factory
# ======================================================================

def lti_dynamics(
    A:  ArrayOrVar,
    B:  ArrayOrVar,
    x0: ArrayOrVar,
    horizon: int,
    c: Optional[ArrayOrVar] = None,
    state_names : Optional[Sequence[str]] = None,
    input_names : Optional[Sequence[str]] = None,
    cst_name    : str = "dyn"
) -> Constraint:
    """Equality constraint for time-invariant affine dynamics.

    Encodes ``x_{k+1} = A x_k + B u_k + c`` for ``k = 0, ..., N-1``.

    Each argument accepts a concrete ``Array`` (constant) or a
    :class:`Variable` (looked up at solve time).

    Parameters
    ----------
    A  : ``(n_x, n_x)`` state matrix.
    B  : ``(n_x, n_u)`` input matrix.
    x0 : ``(n_x,)`` initial state.
    horizon : N (≥ 1).
    c  : ``(n_x,)`` affine term (default zero).
    state_names : Optional[Sequence[str]], default None
        Name of each state, used to construct partitions.
    input_names : Optional[Sequence[str]], default None
        Name of each input, used to construct partitions.
    cst_name : Optional[str], default "dyn"
        A descriptive name of this constraint.

    Returns
    -------
    Constraint (equality)
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")

    N = horizon
    nx = int(A.shape[-1])
    nu = int(B.shape[-1])

    def lhs(v: Dict[str, Array]) -> Array:
        return build_lti_lhs(resolve(A, v), resolve(B, v), N)

    def rhs(v: Dict[str, Array]) -> Array:
        c_val = resolve(c, v) if c is not None else None
        return build_lti_rhs(resolve(A, v), resolve(x0, v), N, c_val)

    v_in_lhs = collect_v_in(A=A, B=B)
    v_in_rhs = collect_v_in(A=A, x0=x0, **({"c": c} if c is not None else {}))

    return Constraint(
        "equality", lhs=lhs, rhs=rhs,
        v_in_lhs=v_in_lhs, v_in_rhs=v_in_rhs,
        var_partition=Partition.state_before_input(
            nx, nu, N,
            state_names=state_names,
            input_names=input_names
        ),
        cst_partition=Partition.dynamics_rows(
            nx, N,
            state_names=state_names,
            cst_name=cst_name
        ),
        name=cst_name
    )


# ======================================================================
# LTV factory
# ======================================================================

def ltv_dynamics(
    A:  ArrayOrVar,
    B:  ArrayOrVar,
    x0: ArrayOrVar,
    horizon: int,
    c: Optional[ArrayOrVar] = None,
    state_names : Optional[Sequence[str]] = None,
    input_names : Optional[Sequence[str]] = None,
    cst_name    : str = "dyn"
) -> Constraint:
    """Equality constraint for time-varying affine dynamics.

    Encodes ``x_{k+1} = A_k x_k + B_k u_k + c_k`` for ``k = 0, ..., N-1``.

    Each argument accepts a concrete ``Array`` (constant) or a
    :class:`Variable` (looked up at solve time).

    Parameters
    ----------
    A  : ``(N, n_x, n_x)`` per-step state matrices.
    B  : ``(N, n_x, n_u)`` per-step input matrices.
    x0 : ``(n_x,)`` initial state.
    horizon : N (≥ 1).
    c  : ``(N, n_x)`` per-step affine terms (default zero).
    state_names : Optional[Sequence[str]], default None
        Name of each state, used to construct partitions.
    input_names : Optional[Sequence[str]], default None
        Name of each input, used to construct partitions.
    cst_name : Optional[str], default "dyn"
        A descriptive name of this constraint.

    Returns
    -------
    Constraint (equality)
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")

    N = horizon
    nx = int(A.shape[-1])
    nu = int(B.shape[-1])

    def lhs(v: Dict[str, Array]) -> Array:
        # A and B are sequence of matrices -> expected ndim = 3
        return build_ltv_lhs(auto_tile(resolve(A, v), N, 3), auto_tile(resolve(B, v), N, 3), N)

    def rhs(v: Dict[str, Array]) -> Array:
        # c is a sequence of vectors -> expected ndim = 2
        c_val = auto_tile(resolve(c, v), N, 2) if c is not None else None
        return build_ltv_rhs(auto_tile(resolve(A, v), N, 3), resolve(x0, v), N, c_val)

    v_in_lhs = collect_v_in(A=A, B=B)
    v_in_rhs = collect_v_in(A=A, x0=x0, **({"c": c} if c is not None else {}))

    return Constraint(
        "equality", lhs=lhs, rhs=rhs,
        v_in_lhs=v_in_lhs, v_in_rhs=v_in_rhs,
        var_partition=Partition.state_before_input(
            nx, nu, N,
            state_names=state_names,
            input_names=input_names
        ),
        cst_partition=Partition.dynamics_rows(
            nx, N,
            state_names=state_names,
            cst_name=cst_name
        ),
        name=cst_name
    )


# ======================================================================
# Nonlinear factory (Linearization)
# ======================================================================

def nonlinear_dynamics(
    dyn         : Dynamics, 
    x_nominal   : ArrayOrVar, 
    u_nominal   : ArrayOrVar, 
    horizon     : int,
    true_params : Optional[Dict[str, ArrayOrVar]] = None,
    state_names : Optional[Sequence[str]] = None,
    input_names : Optional[Sequence[str]] = None,
    cst_name    : str = "dyn"
) -> Constraint:
    """Equality constraint for nonlinear dynamics linearized around a nominal trajectory.

    This automatically differentiates the provided `Dynamics` object using `jax.jacfwd`
    and builds an exact LTV representation along the provided `x_nominal` and `u_nominal` 
    trajectories.


    Parameters
    ----------
    dyn : Dynamics
        The nonlinear dynamics model to linearize.
    x_nominal : ``(N, n_x)`` ArrayOrVar
        The state trajectory to linearize around.
    u_nominal : ``(N, n_u)`` ArrayOrVar
        The input trajectory to linearize around.
    horizon : N (≥ 1).
    true_params : Optional[Dict[str, ArrayOrVar]]
        Parameters passed to the `dyn.step` method.
    state_names : Optional[Sequence[str]], default None
        Name of each state, used to construct partitions.
    input_names : Optional[Sequence[str]], default None
        Name of each input, used to construct partitions.
    cst_name : Optional[str], default "dyn"
        A descriptive name of this constraint.

    Returns
    -------
    Constraint (equality)
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")

    N  = horizon
    nx = int(x_nominal.shape[-1])
    nu = int(u_nominal.shape[-1])

    def linearize_step(x, u, params):
        # A_t = df/dx, B_t = df/du
        A_t = jax.jacfwd(dyn.step, argnums=0)(x, u, params, params)
        B_t = jax.jacfwd(dyn.step, argnums=1)(x, u, params, params)
        
        # c_t = f(x, u) - A_t x - B_t u
        f_val = dyn.step(x, u, params, params)
        c_t = f_val - A_t @ x - B_t @ u
        return A_t, B_t, c_t

    # Vectorize across the trajectory horizon
    linearize_traj = jax.vmap(linearize_step, in_axes=(0, 0, None))

    def lhs(v: Dict[str, Array]) -> Array:
        # x_nominal and u_nominal are trajectories -> expected ndim = 2
        x_nom = auto_tile(resolve(x_nominal, v), N, 2)
        u_nom = auto_tile(resolve(u_nominal, v), N, 2)
        params = {k: resolve(p, v) for k, p in (true_params or {}).items()}
        
        A_traj, B_traj, _ = linearize_traj(x_nom, u_nom, params)
        return build_ltv_lhs(A_traj, B_traj, N)

    def rhs(v: Dict[str, Array]) -> Array:
        x_nom = auto_tile(resolve(x_nominal, v), N, 2)
        u_nom = auto_tile(resolve(u_nominal, v), N, 2)
        params = {k: resolve(p, v) for k, p in (true_params or {}).items()}
        
        A_traj, _, c_traj = linearize_traj(x_nom, u_nom, params)
        return build_ltv_rhs(A_traj, x_nom[0], N, c_traj)

    all_vars = collect_v_in(x_nominal=x_nominal, u_nominal=u_nominal, **(true_params or {}))
    
    return Constraint(
        cst_type="equality",
        lhs=lhs,
        rhs=rhs,
        v_in_lhs=all_vars,
        v_in_rhs=all_vars,
        var_partition=Partition.state_before_input(
            nx, nu, N,
            state_names=state_names,
            input_names=input_names
        ),
        cst_partition=Partition.dynamics_rows(
            nx, N,
            state_names=state_names,
            cst_name=cst_name
        ),
        name=cst_name
    )