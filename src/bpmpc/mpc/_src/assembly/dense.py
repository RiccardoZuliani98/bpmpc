"""Dense-mode QP assembly.

This module implements the dense array assembly backend for the MPC problem.
In dense mode, all matrices and vectors within the :class:`QPData` structure 
are standard contiguous ``jnp.ndarray`` objects.

The dense assembler relies on a pre-allocated ``base_qp`` that is sized to the 
fully expanded decision dimension (``n_dec = n_var + n_slack``). During build-time, 
this base QP is populated with all constant parametric terms and the slack variable 
penalties/couplings.

At runtime, the assembler's ``apply`` method accumulates evaluated parametric terms 
(either slow or fast) directly on top of the base QP using JAX's non-destructive 
``.at[...].add(...)`` scatter operations.
"""

from __future__ import annotations

from typing import (
    Dict, List, NamedTuple, 
    Sequence, Tuple
)

from jax import Array
import jax.numpy as jnp

from bpmpc.mpc._src.types import QPData
from bpmpc.mpc._src.terms import _CostTerm, _CstTerm, Term, decompose_slacks, field_of
from bpmpc.mpc._src.constraint import Constraint
from bpmpc.mpc._src.types import ArrayIn


# ======================================================================
# Assembler
# ======================================================================

class DenseAssembler(NamedTuple):
    """Runtime state and applicator for dense MPC problem assembly.

    Attributes
    ----------
    base_qp : QPData
        The static constant contribution to the problem. It is pre-sized to the 
        fully expanded ``n_dec`` dimension and has the slack penalty matrix, 
        slack linear cost, and slack constraint blocks already folded in.
    slow_terms : List[Term]
        A list of parametric terms that only depend on the declared ``slow_vars``. 
        These are evaluated and applied during the ``prepare()`` phase.
    fast_terms : List[Term]
        A list of parametric terms that depend on fast variables (e.g., current state). 
        These are evaluated and applied during every ``solve()`` call.
    n_var : int
        The original decision dimension (without slack variables).
    n_dec : int
        The total decision dimension (base variables + slacks).
    n_eq : int
        Total number of equality rows in the QP.
    n_ineq : int
        Total number of inequality rows (user constraints + slack non-negativity).
    """
    base_qp:    QPData
    slow_terms: List[Term]
    fast_terms: List[Term]
    n_var:      int
    n_dec:      int
    n_eq:       int
    n_ineq:     int

    def apply(
        self,
        base:  QPData,
        terms: Sequence[Term],
        v:     ArrayIn,
    ) -> QPData:
        """Accumulates evaluated parametric terms into a starting QP.

        Architectural Note
        ------------------
        When this function is compiled with ``jax.jit``, XLA will fully unroll 
        the Python ``for`` loop at compile time. Each term generates a distinct 
        ``ScatterAdd`` node in the computation graph. While this executes extremely 
        fast at runtime, compiling hundreds of individual terms can lead to long 
        JIT compile times. Users are encouraged to "vectorize" mathematically 
        similar constraints into single larger ``Constraint`` blocks where possible.

        Parameters
        ----------
        base : QPData
            The starting QPData state (usually ``self.base_qp`` or a prepared QP).
        terms : Sequence[Term]
            The sequence of atomic terms to evaluate and add.
        v : ArrayIn
            A dictionary mapping variable names to their evaluated JAX arrays.

        Returns
        -------
        QPData
            A newly updated QPData object containing the accumulated terms.
        """
        for t in terms:
            base = _apply(base, t, v)
        return base


# ======================================================================
# Build routines
# ======================================================================

def build(
    const_terms: List[Term],
    slow_terms:  List[Term],
    fast_terms:  List[Term],
    constraints: Sequence[Constraint],
    n_var:       int,
    n_dec:       int,
    n_eq:        int,
    n_ineq:      int,
    n_ineq_user: int,
) -> DenseAssembler:
    """Constructs the DenseAssembler and initializes the base QP state.

    This function executes eagerly at build-time. It allocates the zeroed arrays 
    for the fully expanded QP, evaluates all constant terms, and injects the 
    slack variable configuration.

    Parameters
    ----------
    const_terms : List[Term]
        Terms that require no parameters and can be evaluated immediately.
    slow_terms : List[Term]
        Terms dependent only on slow variables.
    fast_terms : List[Term]
        Terms dependent on fast variables.
    constraints : Sequence[Constraint]
        The raw constraint objects, used here to extract slack variable metadata.
    n_var : int
        The original decision dimension.
    n_dec : int
        The total decision dimension including slacks.
    n_eq : int
        Total equality constraint rows.
    n_ineq : int
        Total inequality constraint rows.
    n_ineq_user : int
        The number of inequality rows belonging to user-defined constraints.

    Returns
    -------
    DenseAssembler
        The fully configured dense assembler ready for runtime evaluation.
    """

    # 1. Allocate an empty, dense QP of the fully expanded size.
    qp = QPData.zeros(n_dec, n_eq, n_ineq)

    # 2. Fold in all constant terms directly (eager evaluation).
    for t in const_terms:
        qp = _apply(qp, t, {})

    # 3. Fold slacks.  Their columns are untouched by the terms above, so
    # scattering the shared coordinate lists in is enough.
    if n_dec > n_var:
        (Pr, Pc, Pv), (Gr, Gc, Gv), q_s = decompose_slacks(constraints, n_var, n_ineq_user)
        qp = qp._replace(
            P=qp.P.at[Pr, Pc].add(Pv),
            q=qp.q.at[n_var:].add(q_s),
            G=qp.G.at[Gr, Gc].add(Gv),
        )

    return DenseAssembler(
        base_qp=qp,
        slow_terms=slow_terms,
        fast_terms=fast_terms,
        n_var=n_var,
        n_dec=n_dec,
        n_eq=n_eq,
        n_ineq=n_ineq,
    )


# ======================================================================
# Internal
# ======================================================================

def _apply(qp: QPData, term: Term, v: ArrayIn) -> QPData:
    """Evaluates ``term`` at ``v`` and accumulates it into ``qp``.

    A term with declared non-zeros (e.g. one returning a ``BCOO``) is
    scatter-added entry by entry straight into its field.  Building its
    dense block first and then adding it into a sub-block of the field
    would cost a dense temporary of the block's size.
    """
    if term.nz is None:
        return _add(qp, term, term.fn(v))
    field = field_of(term)
    rows = term.nz.rows + (term.row_start if isinstance(term, _CstTerm) else 0)
    return qp._replace(**{
        field: getattr(qp, field).at[rows, term.nz.cols].add(term.nz.vals(v))
    })


def _add(qp: QPData, term: Term, val: Array) -> QPData:
    """Adds the evaluated tensor ``val`` into the QPData field implied by ``term``.

    This function maps evaluated cost/constraint blocks into their correct 
    spatial positions within the global QP arrays.

    Crucially, parametric terms (provided by the user) are entirely unaware of 
    slack variables. Therefore, this function strictly writes into the top-left 
    sub-blocks of the matrices (spanning only the base ``n_var`` columns), 
    ensuring that the trailing slack configuration remains undisturbed.

    Parameters
    ----------
    qp : QPData
        The current state of the QP arrays.
    term : Term
        The term descriptor dictating the target field ('P', 'q', 'lhs', etc.) 
        and the relevant row slices.
    val : Array
        The evaluated numerical tensor to add.

    Returns
    -------
    QPData
        A new QPData instance with the updated array field.
    """
    if isinstance(term, _CostTerm):
        # Cost terms only affect the base decision variables (top-left).
        if term.target == "P":
            nr, nc = val.shape
            return qp._replace(P=qp.P.at[:nr, :nc].add(val))
        if term.target == "q":
            return qp._replace(q=qp.q.at[:val.shape[0]].add(val))
        # target == "c"
        return qp._replace(c=qp.c + val)

    # It's a constraint term (_CstTerm)
    rs, re = term.row_start, term.row_end
    
    if term.cst_kind == "eq":
        if term.target == "lhs":
            # Target the specific row block, but only the base variable columns.
            return qp._replace(A=qp.A.at[rs:re, :val.shape[1]].add(val))
        return qp._replace(b=qp.b.at[rs:re].add(val))
        
    # It's an inequality constraint term
    if term.target == "lhs":
        # Target the specific row block, but only the base variable columns.
        return qp._replace(G=qp.G.at[rs:re, :val.shape[1]].add(val))
    return qp._replace(h=qp.h.at[rs:re].add(val))
