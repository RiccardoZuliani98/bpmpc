"""Plumbing shared by :class:`Cost` and :class:`Constraint`.

These helper functions validate and merge variable dictionaries, produce
random sample inputs for probing user callables, and turn callables that
return a ``BCOO`` into a dense callable plus its :class:`NonZeros`.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO

from bpmpc.variable import Variable
from bpmpc.mpc._src.types import ArrayIn, NonZeros


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


# ======================================================================
# BCOO outputs
# ======================================================================

def split_bcoo(
    fn:   Callable[[ArrayIn], BCOO],
    out:  BCOO,
    v_in: Optional[Dict[str, Variable]],
    nz:   Optional[NonZeros],
    what: str,
) -> Tuple[Callable[[ArrayIn], Array], NonZeros, Array]:
    """Splits a callable returning a ``BCOO`` into a dense callable and its non-zeros.

    The BCOO's ``indices`` become the static coordinates and its ``data``
    the values, so the sparse assembler never builds the dense matrix; the
    dense callable serves dense mode and evaluation.  The indices must not
    depend on the inputs: this is checked on two further random samples.
    Padding entries (coordinates outside the shape) are dropped.

    Args:
        fn: The user callable, returning a 2-D ``BCOO``.
        out: ``fn`` evaluated on a sample of ``v_in``.
        v_in: The variables ``fn`` depends on, or ``None`` if constant.
        nz: Non-zeros passed alongside ``fn``; must be ``None``.
        what: Name used in error messages (e.g. ``"q_mat"``).

    Returns:
        The dense callable, the non-zeros, and ``out`` as a dense array.

    Raises:
        ValueError: If ``nz`` is also given, if ``out`` has batch or dense
            dimensions, or if the indices change with the inputs.
    """
    if nz is not None:
        raise ValueError(
            f"{what} returns a BCOO, which already defines its non-zeros; "
            f"do not also pass {what}_nz."
        )
    if out.n_batch or out.n_dense:
        raise ValueError(
            f"{what} must return a BCOO without batch or dense dimensions, got "
            f"n_batch={out.n_batch}, n_dense={out.n_dense}."
        )

    idx = np.asarray(out.indices)
    for _ in range(2 if v_in else 0):
        other = fn(make_sample(v_in))
        if not (isinstance(other, BCOO)
                and np.array_equal(np.asarray(other.indices), idx)):
            raise ValueError(
                f"The BCOO indices returned by {what} change with its inputs. "
                f"They must be static: build them from constants (e.g. NumPy "
                f"arrays), not from the input values."
            )

    rows, cols = idx[:, 0], idx[:, 1]
    keep = (rows < out.shape[0]) & (cols < out.shape[1])
    if keep.all():
        vals = lambda v: fn(v).data
    else:
        rows, cols = rows[keep], cols[keep]
        vals = lambda v: fn(v).data[keep]

    return (lambda v: fn(v).todense()), NonZeros(rows, cols, vals), out.todense()
