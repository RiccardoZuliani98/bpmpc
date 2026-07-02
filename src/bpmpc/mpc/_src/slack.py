"""Slack variable specification for soft constraints.

A ``SlackSpec`` tells the MPC builder which constraint rows to relax
with non-negative slack variables and what constant penalty to apply:

    penalty_i = 0.5 * w_quad_i * s_i**2  +  w_lin_i * s_i

where ``s_i >= 0`` and ``i`` runs over the ``n_slack`` slacked rows.

``rows`` is a binary mask of length ``n_cst`` (1 = slacked, 0 = hard).
``w_quad`` and ``w_lin`` are of length ``n_slack`` (the number of 1s
in ``rows``), one weight per slack variable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class SlackSpec:
    """Which constraint rows are slacked and their penalties.

    Attributes
    ----------
    rows : Tuple[int, ...]
        Binary mask of length ``n_cst``. ``1`` indicates the row is slacked, 
        ``0`` indicates a hard constraint.
    w_quad : Tuple[float, ...]
        Quadratic penalty per slack variable. Length must match ``n_slack``.
    w_lin : Tuple[float, ...]
        Linear penalty per slack variable. Length must match ``n_slack``.
    n_cst : int
        Total number of constraint rows.
    n_slack : int
        Number of slacked rows (i.e., the number of 1s in ``rows``).

    Raises
    ------
    ValueError
        If the length of ``rows`` does not match ``n_cst``.
        If the number of 1s in ``rows`` does not match ``n_slack``.
        If ``rows`` contains values other than 0 or 1.
        If the lengths of ``w_quad`` or ``w_lin`` do not match ``n_slack``.
        If any weights in ``w_quad`` or ``w_lin`` are strictly negative.
    """

    rows:    Tuple[int, ...]
    w_quad:  Tuple[float, ...]
    w_lin:   Tuple[float, ...]
    n_cst:   int
    n_slack: int

    def __post_init__(self) -> None:
        if len(self.rows) != self.n_cst:
            raise ValueError(
                f"rows has length {len(self.rows)} but n_cst={self.n_cst}."
            )
        ones = sum(self.rows)
        if ones != self.n_slack:
            raise ValueError(
                f"rows contains {ones} ones but n_slack={self.n_slack}."
            )
        if not all(r in (0, 1) for r in self.rows):
            raise ValueError("rows must contain only 0s and 1s.")
        if len(self.w_quad) != self.n_slack:
            raise ValueError(
                f"w_quad has length {len(self.w_quad)} but n_slack={self.n_slack}."
            )
        if len(self.w_lin) != self.n_slack:
            raise ValueError(
                f"w_lin has length {len(self.w_lin)} but n_slack={self.n_slack}."
            )
        if any(w < 0 for w in self.w_quad):
            raise ValueError(
                "w_quad contains negative values; penalty weights must be "
                "non-negative."
            )
        if any(w < 0 for w in self.w_lin):
            raise ValueError(
                "w_lin contains negative values; penalty weights must be "
                "non-negative."
            )
        if self.n_slack > 0:
            for i, (wq, wl) in enumerate(zip(self.w_quad, self.w_lin)):
                if wq == 0.0 and wl == 0.0:
                    warnings.warn(
                        f"Slack variable {i} has both w_quad=0 and w_lin=0; "
                        f"the constraint is effectively removed, not softened.",
                        stacklevel=2,
                    )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def slack_rows(
        cls,
        n_cst:  int,
        rows:   Sequence[int],
        w_quad: Union[float, Sequence[float]] = 0.0,
        w_lin:  Union[float, Sequence[float]] = 0.0,
    ) -> "SlackSpec":
        """Slacks specific rows of a constraint with ``n_cst`` rows.

        Parameters
        ----------
        n_cst : int
            The total number of rows in the constraint.
        rows : Sequence[int]
            The specific integer indices of the rows to slack.
        w_quad : Union[float, Sequence[float]], default 0.0
            The quadratic penalty weight(s). If a scalar is provided, it is 
            broadcast to all slacked rows.
        w_lin : Union[float, Sequence[float]], default 0.0
            The linear penalty weight(s). If a scalar is provided, it is 
            broadcast to all slacked rows.

        Returns
        -------
        SlackSpec
            A newly instantiated `SlackSpec` matching the requested configuration.

        Raises
        ------
        ValueError
            If any row index is out of bounds or if there are duplicate row indices.
        """
        row_indices = tuple(rows)
        n_slack     = len(row_indices)

        for r in row_indices:
            if r < 0 or r >= n_cst:
                raise ValueError(
                    f"Slack row index {r} is out of range for n_cst={n_cst}."
                )
        if len(set(row_indices)) != n_slack:
            raise ValueError("Duplicate row indices.")

        mask = [0] * n_cst
        for r in row_indices:
            mask[r] = 1

        wq = _broadcast_weight(w_quad, n_slack, "w_quad")
        wl = _broadcast_weight(w_lin,  n_slack, "w_lin")

        return cls(
            rows=tuple(mask),
            w_quad=wq, w_lin=wl,
            n_cst=n_cst, n_slack=n_slack,
        )

    @classmethod
    def slack_all(
        cls,
        n_cst:  int,
        w_quad: Union[float, Sequence[float]] = 0.0,
        w_lin:  Union[float, Sequence[float]] = 0.0,
    ) -> "SlackSpec":
        """Slacks every row of a constraint.

        Parameters
        ----------
        n_cst : int
            The total number of rows in the constraint.
        w_quad : Union[float, Sequence[float]], default 0.0
            The quadratic penalty weight(s).
        w_lin : Union[float, Sequence[float]], default 0.0
            The linear penalty weight(s).

        Returns
        -------
        SlackSpec
            A `SlackSpec` with all rows configured as soft constraints.
        """
        return cls.slack_rows(n_cst, range(n_cst), w_quad=w_quad, w_lin=w_lin)

    @classmethod
    def slack_none(cls, n_cst: int) -> "SlackSpec":
        """Creates a spec where all constraints are hard (no slack).

        Parameters
        ----------
        n_cst : int
            The total number of rows in the constraint.

        Returns
        -------
        SlackSpec
            An empty `SlackSpec` indicating no rows are slacked.
        """
        return cls.slack_rows(n_cst, [], w_quad=0, w_lin=0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        """bool: True if there are no slacked rows."""
        return self.n_slack == 0

    @property
    def slack_indices(self) -> Tuple[int, ...]:
        """Tuple[int, ...]: The zero-indexed row locations where ``rows[i] == 1``."""
        return tuple(i for i, r in enumerate(self.rows) if r == 1)

    @property
    def rows_array(self) -> Array:
        """Array: The binary row mask as a JAX int32 array."""
        return jnp.array(self.rows, dtype=jnp.int32)

    @property
    def w_quad_array(self) -> Array:
        """Array: The quadratic weights as a JAX array."""
        return jnp.array(self.w_quad)

    @property
    def w_lin_array(self) -> Array:
        """Array: The linear weights as a JAX array."""
        return jnp.array(self.w_lin)

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------
    def add(self, other: "SlackSpec") -> "SlackSpec":
        """Stacks two slack specifications (used for vertically stacked constraints).

        Parameters
        ----------
        other : SlackSpec
            Another `SlackSpec` to append to this one.

        Returns
        -------
        SlackSpec
            A combined `SlackSpec` representing the stacked constraints.
        """
        return SlackSpec(
            rows=self.rows + other.rows,
            w_quad=self.w_quad + other.w_quad,
            w_lin=self.w_lin + other.w_lin,
            n_cst=self.n_cst + other.n_cst,
            n_slack=self.n_slack + other.n_slack,
        )

    def __add__(self, other: "SlackSpec") -> "SlackSpec":
        """Stacks two slack specifications. See ``add`` for details."""
        return self.add(other)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if self.is_empty:
            return f"SlackSpec(n_cst={self.n_cst}, no slack)"
        return (
            f"SlackSpec(n_cst={self.n_cst}, n_slack={self.n_slack}, "
            f"slacked={self.slack_indices})"
        )


# ======================================================================
# Helper
# ======================================================================

def _broadcast_weight(
    w:    Union[float, Sequence[float]],
    n:    int,
    name: str,
) -> Tuple[float, ...]:
    """Expands a scalar to a tuple of length `n`, or validates a sequence's length.

    Parameters
    ----------
    w : Union[float, Sequence[float]]
        The scalar weight to broadcast or sequence to validate.
    n : int
        The target length (number of slack variables).
    name : str
        The variable name to display in error messages (e.g., 'w_quad').

    Returns
    -------
    Tuple[float, ...]
        A tuple of floats representing the weights.

    Raises
    ------
    ValueError
        If ``w`` is a sequence and its length does not equal ``n``.
    """
    if isinstance(w, (int, float)):
        return tuple(float(w) for _ in range(n))
    t = tuple(float(x) for x in w)
    if len(t) != n:
        raise ValueError(f"{name} has length {len(t)} but n_slack={n}.")
    return t