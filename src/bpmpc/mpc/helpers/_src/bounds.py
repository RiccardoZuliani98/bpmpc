"""Box constraint factory and builder.

Two layers:

1. **Builder** (``build_box_lhs``, ``build_box_rhs``) — pure functions
   that take concrete arrays and return the dense matrices.

2. **Factory** (``box_bounds``) — accepts ``Array | Variable`` per
   argument and returns a ready-to-use :class:`Constraint`.

Decision vector layout::

    z = [x_1; ...; x_N; u_0; ...; u_{N-1}]
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import jax.numpy as jnp
from jax import Array

from bpmpc.variable import Variable
from bpmpc.mpc._src.constraint import Constraint
from bpmpc.mpc._src.slack import SlackSpec
from bpmpc.mpc.helpers._src.util import ArrayOrVar, resolve, collect_v_in
from bpmpc.mpc._src.partition import Partition


# ======================================================================
# Builders
# ======================================================================

def build_box_lhs(n_x: int, n_u: int, horizon: int) -> Array:
    """Build the constant LHS for box constraints.

    Returns ``(2*(N*n_x + N*n_u), N*n_x + N*n_u)`` matrix::

        [ Fx; -Fx; Fu; -Fu ]

    where ``Fx = [I | 0]`` selects states and ``Fu = [0 | I]``
    selects inputs.
    """
    N = horizon
    n_x_total = N * n_x
    n_u_total = N * n_u
    Fx = jnp.concatenate([jnp.eye(n_x_total), jnp.zeros((n_x_total, n_u_total))], axis=1)
    Fu = jnp.concatenate([jnp.zeros((n_u_total, n_x_total)), jnp.eye(n_u_total)], axis=1)
    return jnp.concatenate([Fx, -Fx, Fu, -Fu])


def build_box_rhs(
    x_min: Array, x_max: Array,
    u_min: Array, u_max: Array,
    horizon: int,
) -> Array:
    """Build the RHS for box constraints.

    Per-step bounds ``(n_x,)`` / ``(n_u,)`` are tiled over the horizon.

    Returns ``(2*(N*n_x + N*n_u),)`` vector::

        [x_max_tiled; -x_min_tiled; u_max_tiled; -u_min_tiled]
    """
    N = horizon
    return jnp.concatenate([
        jnp.tile(x_max, N), jnp.tile(-x_min, N),
        jnp.tile(u_max, N), jnp.tile(-u_min, N),
    ])


# ======================================================================
# Factory
# ======================================================================

def box_bounds(
    horizon: int,
    u_min: Optional[ArrayOrVar] = None,
    u_max: Optional[ArrayOrVar] = None,
    x_min: Optional[ArrayOrVar] = None,
    x_max: Optional[ArrayOrVar] = None,
    *,
    n_x: Optional[int] = None,
    n_u: Optional[int] = None,
    slack: Optional[SlackSpec] = None,
    state_names: Optional[Sequence[str]] = None,
    input_names: Optional[Sequence[str]] = None,
) -> Constraint:
    """Element-wise state and input box constraints.

    Encodes::

        x_min <= x_t <= x_max,   t = 1, ..., N
        u_min <= u_t <= u_max,   t = 0, ..., N-1

    Each argument accepts a concrete ``Array`` (constant) or a
    :class:`Variable` (looked up at solve time).  Per-step vectors
    ``(n_x,)`` / ``(n_u,)`` are tiled over the horizon.  Concrete
    bounds with ``±inf`` entries are dropped from the constraint set
    (Variable bounds are always kept).

    Parameters
    ----------
    horizon      : N (≥ 1).
    u_min, u_max : ``(n_u,)`` input bounds.
    x_min, x_max : ``(n_x,)`` state bounds.
    n_x          : Optional integer for state dimension if bounds are missing.
    n_u          : Optional integer for input dimension if bounds are missing.
    slack        : Optional :class:`SlackSpec` for soft constraints.

    Returns
    -------
    Constraint (inequality)
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")

    N = horizon
    
    inferred_nx = _dim(x_min, x_max)
    inferred_nu = _dim(u_min, u_max)
    
    n_x_val = n_x if n_x is not None else inferred_nx
    n_u_val = n_u if n_u is not None else inferred_nu

    if n_x_val is None:
        raise ValueError("Could not infer n_x. Please provide n_x explicitly or at least one state bound.")
    if n_u_val is None:
        raise ValueError("Could not infer n_u. Please provide n_u explicitly or at least one input bound.")

    lhs_val = build_box_lhs(n_x_val, n_u_val, N)

    def lhs(_: Dict[str, Array]) -> Array:
        return lhs_val

    def rhs(v: Dict[str, Array]) -> Array:
        _x_min = resolve(x_min, v) if x_min is not None else jnp.full((n_x_val,), -jnp.inf)
        _x_max = resolve(x_max, v) if x_max is not None else jnp.full((n_x_val,), jnp.inf)
        _u_min = resolve(u_min, v) if u_min is not None else jnp.full((n_u_val,), -jnp.inf)
        _u_max = resolve(u_max, v) if u_max is not None else jnp.full((n_u_val,), jnp.inf)
        
        return build_box_rhs(_x_min, _x_max, _u_min, _u_max, N)

    kwargs_to_collect = {}
    if x_min is not None: kwargs_to_collect["x_min"] = x_min
    if x_max is not None: kwargs_to_collect["x_max"] = x_max
    if u_min is not None: kwargs_to_collect["u_min"] = u_min
    if u_max is not None: kwargs_to_collect["u_max"] = u_max

    v_in_rhs = collect_v_in(**kwargs_to_collect)

    # ── drop rows for concretely infinite bounds (Variables are kept) ──
    def _keep(b, d):
        if b is None: return jnp.zeros(d, dtype=bool)
        if isinstance(b, Variable): return jnp.ones(d, dtype=bool)
        return jnp.isfinite(jnp.asarray(b))
    mask = jnp.concatenate([jnp.tile(_keep(b, d), N) for b, d in
        ((x_max, n_x_val), (x_min, n_x_val), (u_max, n_u_val), (u_min, n_u_val))])
    lhs_val = lhs_val[mask]
    _rhs, rhs = rhs, lambda v: _rhs(v)[mask]

    return Constraint(
        "inequality", lhs=lhs, rhs=rhs,
        v_in_rhs=v_in_rhs, slack=slack,
        var_partition=Partition.state_before_input(n_x_val, n_u_val, N, state_names, input_names),
        cst_partition=Partition.box_rows(mask, n_x_val, n_u_val, N),
        name="box"
    )


def _dim(a: Optional[ArrayOrVar], b: Optional[ArrayOrVar]) -> Optional[int]:
    """Infer dimension from whichever argument is concrete, or from the Variable shape."""
    
    def extract_dim(obj) -> int:
        shape = obj.shape
        # Case 1: Shape is a single integer value (e.g., shape = 5)
        if isinstance(shape, int):
            return shape
        # Case 2 & 3: Shape is a tuple/list. 
        # Return the first element, or default to 0 if it's an empty tuple (0-D array).
        return shape[0] if shape else 0

    # First pass: Look for concrete arguments (not Variables)
    for x in (a, b):
        if x is not None and not isinstance(x, Variable):
            return extract_dim(x)
            
    # Second pass: Fallback to Variable shapes
    for x in (a, b):
        if x is not None:
            return extract_dim(x)
            
    return None