"""Quadratic tracking cost factories and builders.

Two layers:

1. **Builders** (``build_*``) — pure functions that take concrete
   arrays and return ``(P, q, c)`` tuples.

2. **Factories** (``state_tracking_cost``, ``output_tracking_cost``)
   — accept ``Array | Variable`` per argument and return a
   ready-to-use :class:`Cost`.

Decision vector layout::

    z = [x_1; ...; x_N; u_0; ...; u_{N-1}]

where ``x_0`` is a parameter, not a decision variable.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Sequence

import jax.numpy as jnp
from jax import Array

from bpmpc.mpc._src.cost import Cost
from bpmpc.mpc.helpers._src.util import ArrayOrVar, resolve, collect_v_in, auto_tile
from bpmpc.mpc._src.partition import Partition


# ======================================================================
# State tracking builder
# ======================================================================

def build_state_tracking(
    Q: Array, R: Array,
    r_x: Array, r_u: Array,
    x0: Array, horizon: int,
) -> Tuple[Array, Array, Array]:
    """Build ``(P, q, c)`` for state/input tracking cost.

    Cost::

        sum_{t=0}^{N-1} ||x_t - r_{x,t}||²_{Q_t} + ||u_t - r_{u,t}||²_{R_t}
          + ||x_N - r_{x,N}||²_{Q_N}

    in ``0.5 z^T P z + q^T z + c`` form.

    Parameters
    ----------
    Q   : ``(N+1, n_x, n_x)`` state weights.
    R   : ``(N, n_u, n_u)`` input weights.
    r_x : ``(N+1, n_x)`` state references.
    r_u : ``(N, n_u)`` input references.
    x0  : ``(n_x,)`` initial state.
    horizon : N.

    Returns
    -------
    P : ``(n_z, n_z)``
    q : ``(n_z,)``
    c : scalar
    """
    N = horizon
    n_x = Q.shape[1]
    n_u = R.shape[1]
    n_x_total = N * n_x
    n_u_total = N * n_u
    n_z = n_x_total + n_u_total
    x0 = x0.reshape((n_x,))

    # Index arrays
    x_idx = jnp.arange(N)[:, None] * n_x + jnp.arange(n_x)[None, :]
    u_idx = n_x_total + jnp.arange(N)[:, None] * n_u + jnp.arange(n_u)[None, :]

    # t=0 state (constant only)
    e0 = x0 - r_x[0]
    c0 = e0 @ (Q[0] @ e0)

    # State blocks t=1..N
    Q_dec = Q[1:]
    r_x_dec = r_x[1:]
    Qr_x = jnp.einsum("tij,tj->ti", Q_dec, r_x_dec)

    # Input blocks t=0..N-1
    Rr_u = jnp.einsum("tij,tj->ti", R, r_u)

    # P: block-diagonal
    P = jnp.zeros((n_z, n_z))
    for t in range(N):
        xi = x_idx[t]
        P = P.at[jnp.ix_(xi, xi)].set(2.0 * Q_dec[t])
        ui = u_idx[t]
        P = P.at[jnp.ix_(ui, ui)].set(2.0 * R[t])

    # q: linear
    q = jnp.zeros(n_z)
    q = q.at[x_idx].add(-2.0 * Qr_x)
    q = q.at[u_idx].add(-2.0 * Rr_u)

    # c: constant
    c = c0 + jnp.einsum("ti,ti->", r_x_dec, Qr_x) + jnp.einsum("ti,ti->", r_u, Rr_u)

    return P, q, c


# ======================================================================
# Output tracking builder
# ======================================================================

def build_output_tracking(
    C: Array, D: Array,
    r: Array, Q: Array,
    x0: Array, horizon: int,
) -> Tuple[Array, Array, Array]:
    """Build ``(P, q, c)`` for output tracking cost.

    Cost::

        sum_{t=0}^{N-1} ||C_t x_t + D_t u_t - r_t||²_{Q_t}
          + ||C_N x_N - r_N||²_{Q_N}

    in ``0.5 z^T P z + q^T z + c`` form.

    Parameters
    ----------
    C  : ``(N+1, n_y, n_x)`` output matrices.
    D  : ``(N, n_y, n_u)`` feedthrough matrices.
    r  : ``(N+1, n_y)`` output references.
    Q  : ``(N+1, n_y, n_y)`` output weights.
    x0 : ``(n_x,)`` initial state.
    horizon : N.

    Returns
    -------
    P : ``(n_z, n_z)``
    q : ``(n_z,)``
    c : scalar
    """
    N = horizon
    T = N + 1
    n_y = C.shape[1]
    n_x = C.shape[2]
    n_u = D.shape[2]
    n_x_total = N * n_x
    n_u_total = N * n_u
    n_z = n_x_total + n_u_total
    n_y_total = T * n_y
    x0 = x0.reshape((n_x,))

    # Index arrays for output mapping matrix
    ks = jnp.arange(T)
    i_y = jnp.arange(n_y)
    i_x = jnp.arange(n_x)
    i_u = jnp.arange(n_u)

    row_base = (ks[:, None, None] * n_y) + i_y[None, :, None]
    rows_C = row_base[1:]
    cols_C = ((ks[1:] - 1)[:, None, None] * n_x) + i_x[None, None, :]
    rows_D = row_base[:-1]
    cols_D = n_x_total + jnp.arange(N)[:, None, None] * n_u + i_u[None, None, :]

    # Constant offsets: d_0 = C_0 x_0 - r_0, d_t = -r_t for t>=1
    d = (-r).at[0].add(C[0] @ x0)

    # Output mapping: y_all = M z + d_all
    M = jnp.zeros((n_y_total, n_z))
    M = M.at[rows_C, cols_C].set(C[1:])
    M = M.at[rows_D, cols_D].set(D)
    Mt = M.reshape((T, n_y, n_z))

    # Batched quadratic form
    QM = jnp.einsum("tij,tjk->tik", Q, Mt)
    P = 2.0 * jnp.einsum("tpi,tpj->ij", Mt, QM)

    Qd = jnp.einsum("tij,tj->ti", Q, d)
    q = 2.0 * jnp.einsum("tpi,tp->i", Mt, Qd)

    c = jnp.einsum("ti,ti->", d, Qd)

    return P, q, c


# ======================================================================
# State tracking factory
# ======================================================================

def state_tracking_cost(
    Q:   ArrayOrVar,
    R:   ArrayOrVar,
    r_x: ArrayOrVar,
    r_u: ArrayOrVar,
    x0:  ArrayOrVar,
    horizon: int,
    state_names: Optional[Sequence[str]] = None,
    input_names: Optional[Sequence[str]] = None,
) -> Cost:
    """Quadratic state/input tracking cost.

    Encodes::

        sum_{t=0}^{N-1} ||x_t - r_{x,t}||²_{Q_t} + ||u_t - r_{u,t}||²_{R_t}
          + ||x_N - r_{x,N}||²_{Q_N}

    Each argument accepts a concrete ``Array`` (constant) or a
    :class:`Variable` (looked up at solve time).

    Parameters
    ----------
    Q           : ``(N+1, n_x, n_x)`` state weights.
    R           : ``(N, n_u, n_u)`` input weights.
    r_x         : ``(N+1, n_x)`` state references.
    r_u         : ``(N, n_u)`` input references.
    x0          : ``(n_x,)`` initial state.
    horizon     : N (≥ 1).
    state_names : Optional[Sequence[str]], default None
        Name of each state, used to construct partitions.
    input_names : Optional[Sequence[str]], default None
        Name of each input, used to construct partitions.

    Returns
    -------
    Cost
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")

    N = horizon
    nx = int(Q.shape[-1])
    nu = int(R.shape[-1])

    def _get_P(v: Dict[str, Array]) -> Array:
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        R_val = auto_tile(resolve(R, v), N, 3)
        n_x = Q_val.shape[1]
        n_u = R_val.shape[1]
        dummy_rx = jnp.zeros((N + 1, n_x))
        dummy_ru = jnp.zeros((N, n_u))
        dummy_x0 = jnp.zeros((n_x,))
        return build_state_tracking(Q_val, R_val, dummy_rx, dummy_ru, dummy_x0, N)[0]

    def _get_q(v: Dict[str, Array]) -> Array:
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        R_val = auto_tile(resolve(R, v), N, 3)
        rx_val = auto_tile(resolve(r_x, v), N + 1, 2)
        ru_val = auto_tile(resolve(r_u, v), N, 2)
        n_x = Q_val.shape[1]
        dummy_x0 = jnp.zeros((n_x,))
        return build_state_tracking(Q_val, R_val, rx_val, ru_val, dummy_x0, N)[1]

    def _get_c(v: Dict[str, Array]) -> Array:
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        R_val = auto_tile(resolve(R, v), N, 3)
        rx_val = auto_tile(resolve(r_x, v), N + 1, 2)
        ru_val = auto_tile(resolve(r_u, v), N, 2)
        x0_val = resolve(x0, v)
        return build_state_tracking(Q_val, R_val, rx_val, ru_val, x0_val, N)[2]

    v_in_q_mat = collect_v_in(Q=Q, R=R)
    v_in_q_vec = collect_v_in(Q=Q, R=R, r_x=r_x, r_u=r_u)
    v_in_c = collect_v_in(Q=Q, R=R, r_x=r_x, r_u=r_u, x0=x0)

    return Cost(
        q_mat=_get_P,
        q_vec=_get_q,
        c=_get_c,
        v_in_q_mat=v_in_q_mat,
        v_in_q_vec=v_in_q_vec,
        v_in_c=v_in_c,
        var_partition=Partition.state_before_input(nx, nu, N, state_names, input_names)
    )


# ======================================================================
# Output tracking factory
# ======================================================================

def output_tracking_cost(
    C:  ArrayOrVar,
    D:  ArrayOrVar,
    r:  ArrayOrVar,
    Q:  ArrayOrVar,
    x0: ArrayOrVar,
    horizon: int,
    state_names: Optional[Sequence[str]] = None,
    input_names: Optional[Sequence[str]] = None,
) -> Cost:
    """Quadratic output tracking cost.

    Encodes::

        sum_{t=0}^{N-1} ||C_t x_t + D_t u_t - r_t||²_{Q_t}
          + ||C_N x_N - r_N||²_{Q_N}

    Each argument accepts a concrete ``Array`` (constant) or a
    :class:`Variable` (looked up at solve time).

    Parameters
    ----------
    C           : ``(N+1, n_y, n_x)`` output matrices.
    D           : ``(N, n_y, n_u)`` feedthrough matrices.
    r           : ``(N+1, n_y)`` output references.
    Q           : ``(N+1, n_y, n_y)`` output weights.
    x0          : ``(n_x,)`` initial state.
    horizon     : N (≥ 1).
    state_names : Optional[Sequence[str]], default None
        Name of each state, used to construct partitions.
    input_names : Optional[Sequence[str]], default None
        Name of each input, used to construct partitions.

    Returns
    -------
    Cost
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")

    N = horizon
    nx = int(C.shape[-1])
    nu = int(D.shape[-1])

    def _get_P(v: Dict[str, Array]) -> Array:
        C_val = auto_tile(resolve(C, v), N + 1, 3)
        D_val = auto_tile(resolve(D, v), N, 3)
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        n_y = C_val.shape[1]
        n_x = C_val.shape[2]
        dummy_r = jnp.zeros((N + 1, n_y))
        dummy_x0 = jnp.zeros((n_x,))
        return build_output_tracking(C_val, D_val, dummy_r, Q_val, dummy_x0, N)[0]

    def _get_q(v: Dict[str, Array]) -> Array:
        C_val = auto_tile(resolve(C, v), N + 1, 3)
        D_val = auto_tile(resolve(D, v), N, 3)
        r_val = auto_tile(resolve(r, v), N + 1, 2)
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        x0_val = resolve(x0, v)
        return build_output_tracking(C_val, D_val, r_val, Q_val, x0_val, N)[1]

    def _get_c(v: Dict[str, Array]) -> Array:
        C_val = auto_tile(resolve(C, v), N + 1, 3)
        D_val = auto_tile(resolve(D, v), N, 3)
        r_val = auto_tile(resolve(r, v), N + 1, 2)
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        x0_val = resolve(x0, v)
        return build_output_tracking(C_val, D_val, r_val, Q_val, x0_val, N)[2]

    v_in_q_mat = collect_v_in(C=C, D=D, Q=Q)
    v_in_q_vec = collect_v_in(C=C, D=D, r=r, Q=Q, x0=x0)
    v_in_c = collect_v_in(C=C, D=D, r=r, Q=Q, x0=x0)

    return Cost(
        q_mat=_get_P,
        q_vec=_get_q,
        c=_get_c,
        v_in_q_mat=v_in_q_mat,
        v_in_q_vec=v_in_q_vec,
        v_in_c=v_in_c,
        var_partition=Partition.state_before_input(nx, nu, N, state_names, input_names)
    )