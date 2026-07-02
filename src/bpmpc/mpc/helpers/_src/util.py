"""Shared utilities for helper factories.

Provides the ``Array``-or-``Variable`` resolution pattern used by
all helper modules, as well as automatic broadcasting.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import Array

from bpmpc.variable import Variable

ArrayOrVar = Union[Array, Variable]


def resolve(arg: ArrayOrVar, v: Dict[str, Array]) -> Array:
    """Return the concrete array for *arg* based on its actual name.

    If *arg* is a :class:`Variable`, look up ``arg.name`` in *v*.
    If *arg* is already an ``Array``, return it directly.
    """
    return v[arg.name] if isinstance(arg, Variable) else arg


def collect_v_in(**kwargs: ArrayOrVar) -> Optional[Dict[str, Variable]]:
    """Collect Variables using their true names as keys.

    Returns ``None`` if no arguments are Variables (fully constant).
    """
    d = {v.name: v for k, v in kwargs.items() if isinstance(v, Variable)}
    return d if d else None


def auto_tile(arr: Array, expected_leading_dim: int, expected_ndim: int) -> Array:
    """Broadcasts an array along the time axis if the time dimension is missing."""
    arr = jnp.asarray(arr)
    if arr.ndim == expected_ndim - 1:
        # Missing the time dimension, prepend the horizon batch dimension
        return jnp.broadcast_to(arr, (expected_leading_dim,) + arr.shape)
    elif arr.ndim == expected_ndim:
        # Already has the time dimension, leave it alone
        return arr
    else:
        raise ValueError(
            f"Expected {expected_ndim} or {expected_ndim - 1} dimensions, "
            f"but got an array with {arr.ndim} dimensions."
        )