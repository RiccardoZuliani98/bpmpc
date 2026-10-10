"""Internal term representations and operations for MPC assembly.

A Cost or Constraint is decomposed into one or more *terms*, each a
(callable, target, variable-set) triple.  Terms are the atomic unit of
work during QP assembly: constant terms are evaluated once at build
time; parametric terms fire at solve time when their dependencies are
satisfied.

This module is mode-agnostic — dense and sparse assembly both consume
the same decomposed terms.
"""

from __future__ import annotations

import uuid

from typing import (
    Callable, Dict, List, Literal, NamedTuple,
    Optional, Sequence, Tuple, TypeVar,
)

import numpy as np
from jax import Array

from bpmpc.variable            import Variable
from bpmpc.mpc._src.cost       import Cost
from bpmpc.mpc._src.constraint import Constraint
from bpmpc.mpc._src.types      import ArrayIn, NonZeros


# ======================================================================
# Term types
# ======================================================================

class _CostTerm(NamedTuple):
    """One callable contribution to the QP cost.

    Attributes
    ----------
    target : Literal["P", "q", "c"]
        The target matrix or vector; 'P' (quadratic), 'q' (linear), or 'c' (constant).
    fn : Callable[[ArrayIn], Array]
        Evaluates the contribution.
    v_in : Optional[Dict[str, Variable]]
        A dictionary of Variables required by `fn`, or `None` if the term is constant.
    uid : str
        A unique identifier auto-generated upon instantiation.
    nz : Optional[NonZeros]
        Optional non-zeros of `fn`'s output, letting the sparse assembler
        skip the dense matrix. `None` means the pattern must be probed.
    """
    target: Literal["P", "q", "c"]
    fn:     Callable[[ArrayIn], Array]
    v_in:   Optional[Dict[str, Variable]]
    uid:    str
    nz:     Optional[NonZeros] = None

    @classmethod
    def create(cls, target, fn, v_in, nz=None):
        """Factory method to auto-generate a UUID on instantiation."""
        return cls(target, fn, v_in, uid=uuid.uuid4().hex, nz=nz)


class _CstTerm(NamedTuple):
    """One callable contribution to a QP constraint.

    Attributes
    ----------
    target : Literal["lhs", "rhs"]
        The target side of the equation; 'lhs' (matrix A or G) or 'rhs' (vector b or h).
    cst_kind : Literal["eq", "ineq"]
        The constraint type; 'eq' (equality → A, b) or 'ineq' (inequality → G, h).
    row_start : int
        The first row index in the global constraint matrix (inclusive).
    row_end : int
        The last row index in the global constraint matrix (exclusive).
    fn : Callable[[ArrayIn], Array]
        Evaluates the contribution.
    v_in : Optional[Dict[str, Variable]]
        A dictionary of Variables required by `fn`, or `None` if the term is constant.
    uid : str
        A unique identifier auto-generated upon instantiation.
    nz : Optional[NonZeros]
        Optional non-zeros of `fn`'s output, with rows local to this term
        (not yet shifted by `row_start`). `None` means the pattern must be
        probed.
    """
    target:    Literal["lhs", "rhs"]
    cst_kind:  Literal["eq", "ineq"]
    row_start: int
    row_end:   int
    fn:        Callable[[ArrayIn], Array]
    v_in:      Optional[Dict[str, Variable]]
    uid:       str
    nz:        Optional[NonZeros] = None

    @classmethod
    def create(cls, target, cst_kind, row_start, row_end, fn, v_in, nz=None):
        """Factory method to auto-generate a UUID on instantiation."""
        return cls(target, cst_kind, row_start, row_end, fn, v_in,
                   uid=uuid.uuid4().hex, nz=nz)


Term = _CostTerm | _CstTerm


def field_of(term: Term) -> str:
    """Determines which QPData field a given term writes to.

    Maps the internal term representation to standard quadratic programming 
    matrices and vectors ('P', 'q', 'c', 'A', 'b', 'G', or 'h').

    Parameters
    ----------
    term : Term
        A `_CostTerm` or `_CstTerm` object.

    Returns
    -------
    str
        A string representing the specific QPData field.
    """
    if isinstance(term, _CostTerm):
        return term.target          # 'P', 'q', or 'c'
    if term.cst_kind == "eq":
        return "A" if term.target == "lhs" else "b"
    return "G" if term.target == "lhs" else "h"


# ======================================================================
# Decomposition
# ======================================================================

def decompose_costs(costs: Sequence[Cost]) -> List[_CostTerm]:
    """Splits a sequence of Cost objects into individual terms.

    Each `Cost` is broken down into three separate `_CostTerm` objects, 
    one for each target field ('P', 'q', and 'c').

    Parameters
    ----------
    costs : Sequence[Cost]
        A sequence of `Cost` objects to decompose.

    Returns
    -------
    List[_CostTerm]
        A list of `_CostTerm` objects representing the decomposed costs.
    """
    terms: List[_CostTerm] = []
    for cost in costs:
        terms.append(_CostTerm.create("P", cost._q_mat, cost.v_in_q_mat, nz=cost._q_mat_nz))
        terms.append(_CostTerm.create("q", cost._q_vec, cost.v_in_q_vec))
        terms.append(_CostTerm.create("c", cost._c,     cost.v_in_c))
    return terms


def decompose_constraints(
    constraints: Sequence[Constraint],
) -> List[_CstTerm]:
    """Splits Constraints into left-hand side (lhs) and right-hand side (rhs) terms.

    Assigns global row offsets to each term based on the constraint's size. 
    Equality and inequality rows are tracked and numbered independently.

    Parameters
    ----------
    constraints : Sequence[Constraint]
        A sequence of `Constraint` objects to decompose.

    Returns
    -------
    List[_CstTerm]
        A list of `_CstTerm` objects containing the decomposed (lhs, rhs) 
        pairs and their corresponding row boundaries.
    """
    terms: List[_CstTerm] = []
    eq_row = ineq_row = 0
    for cst in constraints:
        if cst.is_equality:
            kind: Literal["eq", "ineq"] = "eq"
            rs = eq_row
            eq_row += cst.n_cst
        else:
            kind = "ineq"
            rs = ineq_row
            ineq_row += cst.n_cst
        re = rs + cst.n_cst
        terms.append(_CstTerm.create("lhs", kind, rs, re, cst._lhs, cst.v_in_lhs, nz=cst._lhs_nz))
        terms.append(_CstTerm.create("rhs", kind, rs, re, cst._rhs, cst.v_in_rhs))
    return terms


Coo = Tuple[np.ndarray, np.ndarray, np.ndarray]


def decompose_slacks(
    constraints: Sequence[Constraint],
    n_var:       int,
    n_ineq_user: int,
) -> Tuple[Coo, Coo, np.ndarray]:
    """Coordinates and values of the constant slack contributions.

    Slack ``j`` lives in decision column ``n_var + j``.  It adds its
    quadratic weight on the diagonal of ``P``, its linear weight to ``q``,
    a ``-1`` in ``G`` on the inequality row it relaxes, and a ``-1`` in
    ``G`` on its own non-negativity row ``n_ineq_user + j``.

    Parameters
    ----------
    constraints : Sequence[Constraint]
        The constraints to scan for slack definitions.
    n_var : int
        The number of decision variables before the slacks.
    n_ineq_user : int
        The number of user inequality rows; non-negativity rows follow.

    Returns
    -------
    P : (rows, cols, vals)
        Global coordinates and values of the slack diagonal of ``P``.
    G : (rows, cols, vals)
        Global coordinates and values of the slack columns of ``G``.
    q : np.ndarray
        Linear weights of shape ``(n_slack,)``, for ``q[n_var:]``.
    """
    g_rows, w_quad, w_lin = [], [], []
    row = 0
    for cst in constraints:
        if cst.is_equality:
            continue
        if cst.is_slacked:
            g_rows.extend(row + i for i in cst.slack.slack_indices)
            w_quad.extend(cst.slack.w_quad)
            w_lin.extend(cst.slack.w_lin)
        row += cst.n_cst

    n_slack = len(g_rows)
    cols = n_var + np.arange(n_slack)
    # Multiply by two to counteract the 0.5 imposed by the QP solver.
    P = (cols, cols, 2.0 * np.asarray(w_quad, dtype=float))
    G = (
        np.concatenate([np.asarray(g_rows, dtype=int), n_ineq_user + np.arange(n_slack)]),
        np.concatenate([cols, cols]),
        -np.ones(2 * n_slack),
    )
    return P, G, np.asarray(w_lin, dtype=float)


# ======================================================================
# Variable collection
# ======================================================================

def collect_variables(terms: Sequence[Term]) -> Dict[str, Variable]:
    """Gathers all Variable definitions referenced by a sequence of terms.

    Iterates through the provided terms and extracts required variables, 
    ensuring that variables sharing the same name are identical objects.

    Parameters
    ----------
    terms : Sequence[Term]
        A sequence of `Term` (`_CostTerm` or `_CstTerm`) objects.

    Returns
    -------
    Dict[str, Variable]
        A dictionary mapping variable names to their corresponding `Variable` objects.

    Raises
    ------
    ValueError
        If two terms reference the same name but map to different `Variable` objects.
    """
    out: Dict[str, Variable] = {}
    for t in terms:
        if not t.v_in:
            continue
        for name, var in t.v_in.items():
            if name in out and out[name] != var:
                raise ValueError(
                    f"Conflicting Variable definitions for '{name}'."
                )
            out[name] = var
    return out