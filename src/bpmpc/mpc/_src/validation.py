"""Plumbing shared by :class:`Cost` and :class:`Constraint`.

These helper functions validate and merge variable dictionaries, and produce
random sample inputs for probing user callables.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import jax.numpy as jnp

from bpmpc.variable import Variable
from bpmpc.mpc._src.types import ArrayIn


# ======================================================================
# Variable-dict plumbing
# ======================================================================

def validate_shared_variables(
    vars_a:  Dict[str, Variable],
    vars_b:  Dict[str, Variable],
    label_a: str = "a",
    label_b: str = "b",
) -> None:
    """Validates that overlapping keys in two dictionaries map to identical Variables.

    Checks the names and shapes of variables shared between `vars_a` and `vars_b`.
    If any discrepancies are found, it raises a ValueError to prevent mismatched
    variable definitions from being merged.

    Args:
        vars_a: First dictionary mapping string keys to Variable objects.
        vars_b: Second dictionary mapping string keys to Variable objects.
        label_a: Identifier string for the first dictionary, used in error messages.
        label_b: Identifier string for the second dictionary, used in error messages.

    Raises:
        ValueError: If a shared key maps to Variables with different names or shapes.
    """
    for key in vars_a.keys() & vars_b.keys():
        va, vb = vars_a[key], vars_b[key]
        if va.name != vb.name:
            raise ValueError(
                f"Shared key '{key}': {label_a} name='{va.name}', "
                f"{label_b} name='{vb.name}'."
            )
        if va.shape != vb.shape:
            raise ValueError(
                f"Shared key '{key}' ('{va.name}'): {label_a} "
                f"shape={va.shape}, {label_b} shape={vb.shape}."
            )


def merge_v_in(
    a:       Optional[Dict[str, Variable]],
    b:       Optional[Dict[str, Variable]],
    label_a: str = "a",
    label_b: str = "b",
) -> Optional[Dict[str, Variable]]:
    """Merges two variable dictionaries, ensuring shared keys are identical.

    Combines `a` and `b` into a new dictionary. If keys overlap, they are
    validated using `validate_shared_variables` to ensure consistency before merging.

    Args:
        a: First dictionary of variables, or None if constant/empty.
        b: Second dictionary of variables, or None if constant/empty.
        label_a: Identifier for the first dictionary (used for validation errors).
        label_b: Identifier for the second dictionary (used for validation errors).

    Returns:
        A new dictionary containing the union of variables from `a` and `b`.
        Returns ``None`` only if both `a` and `b` are ``None``.
    """
    if a is None and b is None:
        return None
    if a is None:
        return dict(b)  # type: ignore[arg-type]
    if b is None:
        return dict(a)
    validate_shared_variables(a, b, label_a, label_b)
    return {**a, **b}


# ======================================================================
# Probe sample generation
# ======================================================================

def make_sample(v_in: Optional[Dict[str, Variable]]) -> ArrayIn:
    """Generates random JAX array samples matching the given variable descriptors.

    Creates a dictionary of JAX arrays with shapes corresponding to the
    Variables in `v_in`. Values are drawn from a uniform distribution over [0, 1)
    using NumPy's random number generator and then converted to JAX arrays.

    Args:
        v_in: A dictionary mapping string keys to Variable objects, or None.

    Returns:
        A dictionary mapping the same keys to JAX arrays of random numbers.
        Returns an empty dictionary ``{}`` if `v_in` is ``None``.
    """
    if v_in is None:
        return {}
    return {
        key: jnp.asarray(np.random.rand(*var.shape))
        for key, var in v_in.items()
    }