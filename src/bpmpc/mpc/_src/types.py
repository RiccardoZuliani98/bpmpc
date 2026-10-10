"""Internal data types shared across the package.

* :data:`ArrayIn`   — the ``dict[str, Array]`` input to user callables.
* :data:`SparseMode` — ``'dense'`` | ``'sparse'``.
* :data:`QArray`    — dense :class:`jax.Array` or sparse :class:`BCOO`.
* :class:`QPData`   — the seven arrays defining a convex QP.
* :class:`SlackData`— build-time constants describing slack penalties.
* :class:`NonZeros` — static coordinates and values of a matrix's non-zeros.
"""

from __future__ import annotations

from typing import Callable, Dict, Literal, NamedTuple, Union

import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO


# ======================================================================
# Basic aliases
# ======================================================================

#: Input dictionary passed to every user callable.
ArrayIn = Dict[str, Array]

#: Assembly mode for QPData storage.
SparseMode = Literal["dense", "sparse"]

#: Dense or sparse matrix/vector.
QArray = Union[Array, BCOO]


# ======================================================================
# QPData
# ======================================================================

class QPData(NamedTuple):
    """Numeric arrays defining a convex QP.

    All fields are ``jnp.ndarray`` in dense mode.  In sparse mode the
    matrix fields ``P``, ``A``, ``G`` become ``BCOO`` at the end of
    assembly; the vector fields stay dense.

    Attributes
    ----------
    P : (n, n)      quadratic cost (positive semi-definite).
    q : (n,)        linear cost.
    c : ()          constant cost offset.
    A : (m_eq, n)   equality constraint matrix.
    b : (m_eq,)     equality constraint rhs.
    G : (m_ineq, n) inequality constraint matrix.
    h : (m_ineq,)   inequality constraint rhs.
    """

    P: QArray
    q: QArray
    c: QArray
    A: QArray
    b: QArray
    G: QArray
    h: QArray

    @staticmethod
    def zeros(n_var: int, n_eq: int, n_ineq: int) -> "QPData":
        """All-zero dense QPData with the given dimensions."""
        z = jnp.zeros
        return QPData(
            P=z((n_var, n_var)), q=z((n_var,)), c=z(()),
            A=z((n_eq,  n_var)), b=z((n_eq,)),
            G=z((n_ineq, n_var)), h=z((n_ineq,)),
        )

    def __add__(self, other: "QPData") -> "QPData":
        """Element-wise addition (assumes matching shapes)."""
        return QPData(*(a + b for a, b in zip(self, other)))


# ======================================================================
# SlackData
# ======================================================================

class SlackData(NamedTuple):
    """Precomputed constant arrays for slack variable expansion.

    Built once at :meth:`MPCProblem.__init__`.  Always dense (in 
    sparse mode this is later turned into a sparse construct).

    Attributes
    ----------
    P_slack      : (n_slack, n_slack) diagonal quadratic penalty.
    q_slack      : (n_slack,)         linear penalty.
    G_slack_cols : (n_ineq, n_slack)  columns appended to G.
    n_slack      : total number of slack variables.
    """

    P_slack:      Array
    q_slack:      Array
    G_slack_cols: Array
    n_slack:      int

    @property
    def is_empty(self) -> bool:
        return self.n_slack == 0

    @staticmethod
    def empty(n_ineq: int) -> "SlackData":
        """SlackData with zero slack variables."""
        return SlackData(
            jnp.zeros((0, 0)),
            jnp.zeros((0,)),
            jnp.zeros((n_ineq, 0)),
            0,
        )


# ======================================================================
# NonZeros
# ======================================================================

class NonZeros(NamedTuple):
    """Coordinate (COO) description of a matrix's structural non-zeros.

    Internal: derived from a cost / constraint callable that returns a
    ``BCOO``.  Lets the assemblers write a term entry by entry without
    building its dense matrix.  Coordinates may repeat; repeated entries
    add up.

    Attributes
    ----------
    rows : np.ndarray
        Static 1-D row indices, local to the term (row 0 is its first row).
    cols : np.ndarray
        Static 1-D column indices, same length as ``rows``.
    vals : Callable[[ArrayIn], Array]
        Returns the 1-D values at ``(rows, cols)``, in the same order.
    """

    rows: np.ndarray
    cols: np.ndarray
    vals: Callable[[ArrayIn], Array]

    def append(self, other: "NonZeros", row_shift: int = 0) -> "NonZeros":
        """Concatenate ``other``'s entries after ours, shifting its rows."""
        va, vb = self.vals, other.vals
        return NonZeros(
            rows=np.concatenate([self.rows, other.rows + row_shift]),
            cols=np.concatenate([self.cols, other.cols]),
            vals=lambda v: jnp.concatenate([va(v), vb(v)]),
        )
