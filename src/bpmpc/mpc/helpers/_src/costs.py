"""Quadratic tracking cost factories and builders.

Two layers:

1. **Builders** (``build_*``) — pure functions that take concrete
   arrays and return ``(P, q, c)`` tuples.  ``P`` is a ``BCOO`` with
   static indices by default, or dense when ``sparse=False``.

2. **Factories** (``state_tracking_cost``, ``output_tracking_cost``)
   — accept ``Array | Variable`` per argument and return a
   ready-to-use :class:`Cost`.

Decision vector layout::

    z = [x_1; ...; x_N; u_0; ...; u_{N-1}]

where ``x_0`` is a parameter, not a decision variable.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Sequence, Union

import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO

from bpmpc.mpc._src.cost import Cost
from bpmpc.mpc.helpers._src.util import ArrayOrVar, resolve, collect_v_in, auto_tile, coo_indices
from bpmpc.mpc._src.partition import Partition


# ======================================================================
# State tracking builder
# ======================================================================

def _decision_indices(n_x: int, n_u: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Index grids of the per-step state and input blocks of ``z``.

    Returns ``(x_idx, u_idx, n_z)`` where ``x_idx[t]`` holds the decision
    indices of ``x_{t+1}`` and ``u_idx[t]`` those of ``u_t``.
    """
    N = horizon
    n_x_total = N * n_x
    ks = np.arange(N)[:, None]
    x_idx = ks * n_x + np.arange(n_x)[None, :]
    u_idx = n_x_total + ks * n_u + np.arange(n_u)[None, :]
    return x_idx, u_idx, n_x_total + N * n_u


def build_state_tracking_mat(
    Q: Array, R: Array, horizon: int, *, sparse: bool = True,
) -> Union[Array, BCOO]:
    """Build only ``P`` for the state/input tracking cost.

    See :func:`build_state_tracking`.  Split out so callers that need the
    quadratic term alone do not trace the reference-dependent work.
    """
    P = _state_tracking_mat_bcoo(Q, R, horizon)
    return P if sparse else P.todense()


def _state_tracking_mat_bcoo(Q: Array, R: Array, horizon: int) -> BCOO:
    """The state-tracking ``P`` as a ``BCOO`` with static indices.

    Block-diagonal: the per-step state blocks, then the per-step input
    blocks.  Each family is one batch of entries, so trace and compile
    time do not grow with a Python loop over the horizon.
    """
    x_idx, u_idx, n_z = _decision_indices(Q.shape[1], R.shape[1], horizon)
    indices = coo_indices(
        (x_idx[:, :, None], x_idx[:, None, :]),
        (u_idx[:, :, None], u_idx[:, None, :]),
    )
    data = 2.0 * jnp.concatenate([Q[1:].ravel(), R.ravel()])
    return BCOO((data, indices), shape=(n_z, n_z))


def build_state_tracking_vec(
    Q: Array, R: Array,
    r_x: Array, r_u: Array,
    horizon: int,
) -> Array:
    """Build only ``q`` for the state/input tracking cost.

    See :func:`build_state_tracking`.
    """
    x_idx, u_idx, n_z = _decision_indices(Q.shape[1], R.shape[1], horizon)
    Qr_x = jnp.einsum("tij,tj->ti", Q[1:], r_x[1:])
    Rr_u = jnp.einsum("tij,tj->ti", R, r_u)

    q = jnp.zeros(n_z)
    q = q.at[x_idx].add(-2.0 * Qr_x)
    return q.at[u_idx].add(-2.0 * Rr_u)


def build_state_tracking_const(
    Q: Array, R: Array,
    r_x: Array, r_u: Array,
    x0: Array, horizon: int,
) -> Array:
    """Build only ``c`` for the state/input tracking cost.

    See :func:`build_state_tracking`.
    """
    x0 = x0.reshape((Q.shape[1],))

    # t=0 state contributes a constant only (x_0 is a parameter).
    e0 = x0 - r_x[0]
    c0 = e0 @ (Q[0] @ e0)

    Qr_x = jnp.einsum("tij,tj->ti", Q[1:], r_x[1:])
    Rr_u = jnp.einsum("tij,tj->ti", R, r_u)
    return c0 + jnp.einsum("ti,ti->", r_x[1:], Qr_x) + jnp.einsum("ti,ti->", r_u, Rr_u)


def build_state_tracking(
    Q: Array, R: Array,
    r_x: Array, r_u: Array,
    x0: Array, horizon: int,
    *, sparse: bool = True,
) -> Tuple[Union[Array, BCOO], Array, Array]:
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
    sparse : if True (default), return ``P`` as a ``BCOO`` with static
        indices, so a ``Cost`` built from it skips probing for the
        sparsity pattern.  If False, return a dense array.

    Returns
    -------
    P : ``(n_z, n_z)``
    q : ``(n_z,)``
    c : scalar
    """
    return (
        build_state_tracking_mat(Q, R, horizon, sparse=sparse),
        build_state_tracking_vec(Q, R, r_x, r_u, horizon),
        build_state_tracking_const(Q, R, r_x, r_u, x0, horizon),
    )


# ======================================================================
# Output tracking builder
# ======================================================================

def _output_offsets(C: Array, r: Array, x0: Array) -> Array:
    """Constant part of each output block: ``d_0 = C_0 x_0 - r_0``, ``d_t = -r_t``."""
    x0 = x0.reshape((C.shape[2],))
    return (-r).at[0].add(C[0] @ x0)


def build_output_tracking_mat(
    C: Array, D: Array, Q: Array, horizon: int, *, sparse: bool = True,
) -> Union[Array, BCOO]:
    """Build only ``P`` for the output tracking cost.

    The output map ``y = M z + d`` is block-sparse: block ``t`` touches only
    ``x_t`` and ``u_t``.  Forming ``M`` densely and contracting
    ``M^T Q M`` over the full decision dimension costs ``O(N^3)``; here the
    four families of non-zero blocks are contracted directly, which is
    ``O(N)``.  See :func:`build_output_tracking`.
    """
    P = _output_tracking_mat_bcoo(C, D, Q, horizon)
    return P if sparse else P.todense()


def _output_tracking_mat_bcoo(C: Array, D: Array, Q: Array, horizon: int) -> BCOO:
    """The output-tracking ``P`` as a ``BCOO`` with static indices.

    Stores the four non-zero block families: state diagonal, input
    diagonal and the two state/input cross blocks.
    """
    N = horizon
    x_idx, u_idx, n_z = _decision_indices(C.shape[2], D.shape[2], N)

    C_dec = C[1:]                                        # C_t for t = 1..N
    QC = jnp.einsum("tij,tjk->tik", Q[1:], C_dec)        # Q_t C_t,  t = 1..N
    QD = jnp.einsum("tij,tjk->tik", Q[:-1], D)           # Q_t D_t,  t = 0..N-1

    # (x_t, x_t) for t = 1..N and (u_t, u_t) for t = 0..N-1
    Pxx = jnp.einsum("tpi,tpj->tij", C_dec, QC)
    Puu = jnp.einsum("tpi,tpj->tij", D, QD)

    # Cross blocks exist only where a single output block sees both
    # x_t and u_t, i.e. t = 1..N-1.  Both orderings are formed explicitly
    # so that a non-symmetric Q behaves as it did before.
    QxD = jnp.einsum("tij,tjk->tik", Q[1:N], D[1:])
    Pxu = jnp.einsum("tpi,tpk->tik", C_dec[:-1], QxD)
    Pux = jnp.einsum("tpk,tpi->tki", D[1:], QC[:-1])

    indices = coo_indices(
        (x_idx[:, :, None], x_idx[:, None, :]),
        (u_idx[:, :, None], u_idx[:, None, :]),
        (x_idx[:-1, :, None], u_idx[1:, None, :]),
        (u_idx[1:, :, None], x_idx[:-1, None, :]),
    )
    data = 2.0 * jnp.concatenate([Pxx.ravel(), Puu.ravel(), Pxu.ravel(), Pux.ravel()])
    return BCOO((data, indices), shape=(n_z, n_z))


def build_output_tracking_vec(
    C: Array, D: Array,
    r: Array, Q: Array,
    x0: Array, horizon: int,
) -> Array:
    """Build only ``q`` for the output tracking cost.

    See :func:`build_output_tracking`.
    """
    N = horizon
    n_x = C.shape[2]
    n_u = D.shape[2]
    x_idx, u_idx, n_z = _decision_indices(n_x, n_u, N)

    d = _output_offsets(C, r, x0)
    Qd = jnp.einsum("tij,tj->ti", Q, d)

    q = jnp.zeros(n_z)
    q = q.at[x_idx].add(2.0 * jnp.einsum("tpi,tp->ti", C[1:], Qd[1:]))
    return q.at[u_idx].add(2.0 * jnp.einsum("tpk,tp->tk", D, Qd[:-1]))


def build_output_tracking_const(
    C: Array, r: Array, Q: Array, x0: Array,
) -> Array:
    """Build only ``c`` for the output tracking cost.

    See :func:`build_output_tracking`.
    """
    d = _output_offsets(C, r, x0)
    return jnp.einsum("ti,ti->", d, jnp.einsum("tij,tj->ti", Q, d))


def build_output_tracking(
    C: Array, D: Array,
    r: Array, Q: Array,
    x0: Array, horizon: int,
    *, sparse: bool = True,
) -> Tuple[Union[Array, BCOO], Array, Array]:
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
    sparse : return ``P`` as a ``BCOO`` (default) or dense; see
        :func:`build_state_tracking`.

    Returns
    -------
    P : ``(n_z, n_z)``
    q : ``(n_z,)``
    c : scalar
    """
    return (
        build_output_tracking_mat(C, D, Q, horizon, sparse=sparse),
        build_output_tracking_vec(C, D, r, Q, x0, horizon),
        build_output_tracking_const(C, r, Q, x0),
    )


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

    def _get_P(v: Dict[str, Array]) -> BCOO:
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        R_val = auto_tile(resolve(R, v), N, 3)
        return _state_tracking_mat_bcoo(Q_val, R_val, N)

    def _get_q(v: Dict[str, Array]) -> Array:
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        R_val = auto_tile(resolve(R, v), N, 3)
        rx_val = auto_tile(resolve(r_x, v), N + 1, 2)
        ru_val = auto_tile(resolve(r_u, v), N, 2)
        return build_state_tracking_vec(Q_val, R_val, rx_val, ru_val, N)

    def _get_c(v: Dict[str, Array]) -> Array:
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        R_val = auto_tile(resolve(R, v), N, 3)
        rx_val = auto_tile(resolve(r_x, v), N + 1, 2)
        ru_val = auto_tile(resolve(r_u, v), N, 2)
        x0_val = resolve(x0, v)
        return build_state_tracking_const(Q_val, R_val, rx_val, ru_val, x0_val, N)

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

    def _get_P(v: Dict[str, Array]) -> BCOO:
        C_val = auto_tile(resolve(C, v), N + 1, 3)
        D_val = auto_tile(resolve(D, v), N, 3)
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        return _output_tracking_mat_bcoo(C_val, D_val, Q_val, N)

    def _get_q(v: Dict[str, Array]) -> Array:
        C_val = auto_tile(resolve(C, v), N + 1, 3)
        D_val = auto_tile(resolve(D, v), N, 3)
        r_val = auto_tile(resolve(r, v), N + 1, 2)
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        x0_val = resolve(x0, v)
        return build_output_tracking_vec(C_val, D_val, r_val, Q_val, x0_val, N)

    def _get_c(v: Dict[str, Array]) -> Array:
        C_val = auto_tile(resolve(C, v), N + 1, 3)
        r_val = auto_tile(resolve(r, v), N + 1, 2)
        Q_val = auto_tile(resolve(Q, v), N + 1, 3)
        x0_val = resolve(x0, v)
        return build_output_tracking_const(C_val, r_val, Q_val, x0_val)

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