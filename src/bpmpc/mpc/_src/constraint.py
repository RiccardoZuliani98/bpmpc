"""Constraint descriptor for parametric MPC problems.

A constraint has the form:

    LHS(v) @ z  {==, <=}  RHS(v)

Either side may be constant (``v_in_lhs=None`` or ``v_in_rhs=None``).
When a side is constant the corresponding callable must accept an
empty dict ``{}`` so the calling convention is uniform.
"""

from __future__ import annotations

from typing import Callable, Dict, Literal, Optional, Sequence, Union

import jax.numpy as jnp
from jax import Array

from bpmpc.variable             import Variable
from bpmpc.mpc._src.slack       import SlackSpec
from bpmpc.mpc._src.types       import ArrayIn, NonZeros
from bpmpc.mpc._src.validation  import validate_shared_variables, make_sample, merge_v_in
from bpmpc.mpc._src.partition   import Partition


class Constraint:
    """Parametric linear constraint ``LHS(v) @ z {==, <=} RHS(v)``.

    Parameters
    ----------
    cst_type : Literal["equality", "inequality"]
        The type of constraint ('equality' or 'inequality').
    lhs : Union[Array, Callable[[ArrayIn], Array]]
        Callable returning the left-hand side matrix of shape ``(n_cst, n_var)``, or a constant array.
    rhs : Union[Array, Callable[[ArrayIn], Array]]
        Callable returning the right-hand side vector of shape ``(n_cst,)``, or a constant array.
    v_in_lhs : Optional[Dict[str, Variable]], default None
        Dictionary of variables the LHS depends on, or ``None`` if constant.
    v_in_rhs : Optional[Dict[str, Variable]], default None
        Dictionary of variables the RHS depends on, or ``None`` if constant.
    slack : Optional[SlackSpec], default None
        Optional slack specification. Only inequality constraints may be slacked;
        ``slack.n_cst`` must match ``n_cst``.
    cst_partition : Optional[Partition], default None
        Description of each constraint using the "Partition" class.
    var_partition : Optional[Partition], default None
        Description of each variable using the "Partition" class.
    lhs_nz : Optional[NonZeros], default None
        Optional coordinate description of ``lhs``'s non-zeros.  When given,
        the sparse assembler writes only these entries instead of
        materialising the dense ``(n_cst, n_var)`` matrix.

    Attributes
    ----------
    cst_type : Literal["equality", "inequality"]
        The type of constraint.
    v_in_lhs : Optional[Dict[str, Variable]]
        Variables the left-hand side matrix depends on.
    v_in_rhs : Optional[Dict[str, Variable]]
        Variables the right-hand side vector depends on.
    slack : Optional[SlackSpec]
        The slack specification, if provided.
    n_cst : int
        The number of rows in the constraint.
    n_var : int
        The number of decision variables (columns) in the LHS matrix.
    cst_partition : Optional[Partition]
        Description of each constraint using the "Partition" class.
    var_partition : Optional[Partition]
        Description of each variable using the "Partition" class.
    name : Optional[str]
        Descriptive name of the constraint.

    Raises
    ------
    ValueError
        If slack is provided for an equality constraint.
        If ``slack.n_cst`` does not match the LHS row count.
        If ``lhs`` or ``rhs`` return arrays of incorrect dimension or mismatched sizes.
    """

    cst_type:       Literal["equality", "inequality"]
    v_in_lhs:       Optional[Dict[str, Variable]]
    v_in_rhs:       Optional[Dict[str, Variable]]
    slack_spec:     Optional[SlackSpec]
    n_cst:          int
    n_var:          int
    cst_partition:  Optional[Partition]
    var_partition:  Optional[Partition]
    name:           Optional[str]
    lhs_nz:         Optional[NonZeros]

    def __init__(
        self,
        cst_type:       Literal["equality", "inequality"],
        lhs:            Union[Array, Callable[[ArrayIn], Array]],
        rhs:            Union[Array, Callable[[ArrayIn], Array]],
        v_in_lhs:       Optional[Dict[str, Variable]] = None,
        v_in_rhs:       Optional[Dict[str, Variable]] = None,
        slack:          Optional[SlackSpec] = None,
        cst_partition:  Optional[Partition] = None,
        var_partition:  Optional[Partition] = None,
        name:           Optional[str] = None,
        lhs_nz:         Optional[NonZeros] = None,
    ) -> None:
        self.cst_type = cst_type
        self.v_in_lhs = v_in_lhs
        self.v_in_rhs = v_in_rhs

        if v_in_lhs is not None and v_in_rhs is not None:
            validate_shared_variables(v_in_lhs, v_in_rhs, "lhs", "rhs")

        if callable(lhs):
            self._lhs = lhs
        else:
            _lhs_arr = jnp.asarray(lhs)
            self._lhs = lambda _: _lhs_arr
            
        if callable(rhs):
            self._rhs = rhs
        else:
            _rhs_arr = jnp.asarray(rhs)
            self._rhs = lambda _: _rhs_arr

        lhs_sample = make_sample(v_in_lhs)
        lhs_num = self._lhs(lhs_sample)
        rhs_num = self._rhs(make_sample(v_in_rhs))
        self._validate_shapes(lhs_num, rhs_num)

        if lhs_nz is not None:
            lhs_nz.check(lhs_num.shape, lhs_sample, "lhs_nz")
        self.lhs_nz = lhs_nz

        self.n_cst = lhs_num.shape[0]
        self.n_var = lhs_num.shape[1]

        self.cst_partition = cst_partition
        self.var_partition = var_partition
        self.name = name

        if slack is not None:
            if self.is_equality:
                raise ValueError(
                    "Slack variables are only supported on inequality "
                    "constraints."
                )
            if slack.n_cst != self.n_cst:
                raise ValueError(
                    f"SlackSpec has n_cst={slack.n_cst} but the constraint "
                    f"has n_cst={self.n_cst}."
                )
        self.slack = slack

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_equality(self) -> bool:
        """bool: True if this is an equality constraint."""
        return self.cst_type == "equality"

    @property
    def is_parametric_lhs(self) -> bool:
        """bool: True if the left-hand side matrix depends on runtime variables."""
        return self.v_in_lhs is not None and len(self.v_in_lhs) > 0

    @property
    def is_parametric_rhs(self) -> bool:
        """bool: True if the right-hand side vector depends on runtime variables."""
        return self.v_in_rhs is not None and len(self.v_in_rhs) > 0

    @property
    def is_parametric(self) -> bool:
        """bool: True if either the LHS or RHS depends on runtime variables."""
        return self.is_parametric_lhs or self.is_parametric_rhs

    @property
    def is_slacked(self) -> bool:
        """bool: True if a non-empty SlackSpec is attached to this constraint."""
        return self.slack is not None and not self.slack.is_empty

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def eval_lhs(self, v: Optional[ArrayIn] = None) -> Array:
        """Evaluates the left-hand side matrix.

        Parameters
        ----------
        v : Optional[ArrayIn], default None
            Dictionary of runtime array inputs.

        Returns
        -------
        Array
            The evaluated LHS matrix of shape ``(n_cst, n_var)``.
        """
        return self._lhs(v if v is not None else {})

    def eval_rhs(self, v: Optional[ArrayIn] = None) -> Array:
        """Evaluates the right-hand side vector.

        Parameters
        ----------
        v : Optional[ArrayIn], default None
            Dictionary of runtime array inputs.

        Returns
        -------
        Array
            The evaluated RHS vector of shape ``(n_cst,)``.
        """
        return self._rhs(v if v is not None else {})

    # ------------------------------------------------------------------
    # Modifiers
    # ------------------------------------------------------------------
    def add_slack(
        self,
        rows:   Optional[Sequence[int]] = None,
        w_quad: Union[float, Sequence[float]] = 0.0,
        w_lin:  Union[float, Sequence[float]] = 0.0,
    ) -> "Constraint":
        """Returns a new constraint with slack variables attached.

        Parameters
        ----------
        rows : Optional[Sequence[int]], default None
            The specific integer indices of the rows to slack. If ``None``,
            all rows are slacked.
        w_quad : Union[float, Sequence[float]], default 0.0
            The quadratic penalty weight(s).
        w_lin : Union[float, Sequence[float]], default 0.0
            The linear penalty weight(s).

        Returns
        -------
        Constraint
            A newly constructed Constraint with the requested slack specification.
            
        Raises
        ------
        ValueError
            If this is an equality constraint, which cannot be slacked.
        """
        if self.is_equality:
            raise ValueError(
                "Slack variables are only supported on inequality constraints."
            )
            
        if rows is None:
            new_slack = SlackSpec.slack_all(self.n_cst, w_quad=w_quad, w_lin=w_lin)
        else:
            new_slack = SlackSpec.slack_rows(self.n_cst, rows, w_quad=w_quad, w_lin=w_lin)
            
        return Constraint(
            cst_type=self.cst_type,
            lhs=self._lhs,
            rhs=self._rhs,
            v_in_lhs=self.v_in_lhs,
            v_in_rhs=self.v_in_rhs,
            slack=new_slack,
            var_partition=self.var_partition,
            cst_partition=self.cst_partition,
            name=self.name,
            lhs_nz=self.lhs_nz,
        )

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------
    def add(self, other: "Constraint") -> "Constraint":
        """Vertically stacks two constraints.

        Parameters
        ----------
        other : Constraint
            Another Constraint object to stack below this one.

        Returns
        -------
        Constraint
            A newly constructed Constraint representing the vertically stacked equations.

        Raises
        ------
        ValueError
            If the constraint types ('equality' vs 'inequality') do not match.
            If the decision variable counts (``n_var``) do not match.
        """
        if self.cst_type != other.cst_type:
            raise ValueError(
                f"Cannot add constraints of different types: "
                f"'{self.cst_type}' vs '{other.cst_type}'."
            )
        if self.n_var != other.n_var:
            raise ValueError(
                f"Cannot add constraints with different n_var: "
                f"{self.n_var} vs {other.n_var}."
            )

        merged_lhs = merge_v_in(self.v_in_lhs, other.v_in_lhs,
                                "self.lhs", "other.lhs")
        merged_rhs = merge_v_in(self.v_in_rhs, other.v_in_rhs,
                                "self.rhs", "other.rhs")

        if self.slack is None and other.slack is None:
            merged_slack = None
        else:
            s_self  = self.slack  or SlackSpec.slack_none(self.n_cst)
            s_other = other.slack or SlackSpec.slack_none(other.n_cst)
            merged_slack = s_self + s_other

        # merge constraint partition
        merged_cst_partition = None
        if self.cst_partition and other.cst_partition:
            # Offset the second constraint's partitions by the first constraint's row count
            offset_other = other.cst_partition.offset(self.n_cst)
            merged_mapping = {**self.cst_partition._mapping, **offset_other._mapping}
            merged_cst_partition = Partition(merged_mapping)
        elif self.cst_partition:
            merged_cst_partition = self.cst_partition
        elif other.cst_partition:
            merged_cst_partition = other.cst_partition.offset(self.n_cst)

        # ensure variable partition matches
        merged_var_partition = None
        if self.var_partition and other.var_partition:
            merged_mapping = dict(self.var_partition._mapping)
            for key, val in other.var_partition._mapping.items():
                if key in merged_mapping:
                    val_self = merged_mapping[key]
                    # Check for exact equality to prevent conflicting column layouts
                    if isinstance(val, slice) and isinstance(val_self, slice):
                        if (val.start, val.stop, val.step) != (val_self.start, val_self.stop, val_self.step):
                            raise ValueError(f"var_partition conflict on key '{key}' during constraint addition.")
                    else:
                        # Fallback for array comparisons
                        import numpy as np
                        if not np.array_equal(np.asarray(val), np.asarray(val_self)):
                            raise ValueError(f"var_partition conflict on array key '{key}' during constraint addition.")
                else:
                    merged_mapping[key] = val
            merged_var_partition = Partition(merged_mapping)
        elif self.var_partition:
            merged_var_partition = self.var_partition
        elif other.var_partition:
            merged_var_partition = other.var_partition

        lhs_a, lhs_b = self._lhs, other._lhs
        rhs_a, rhs_b = self._rhs, other._rhs

        # Stacking appends the lower block's non-zeros, shifted down by this
        # constraint's height.  Only available when both halves carry them.
        merged_nz = None
        if self.lhs_nz is not None and other.lhs_nz is not None:
            merged_nz = self.lhs_nz.append(other.lhs_nz, row_shift=self.n_cst)

        names = set((elem for elem in [self.name, other.name] if elem is not None))

        if len(names) == 1: 
            name = names.pop()
        elif len(names) == 2:
            name = names.pop() + "_and_" + names.pop()
        else:
            name = None
        
        return Constraint(
            cst_type=self.cst_type,
            lhs=lambda v: jnp.concatenate([lhs_a(v), lhs_b(v)], axis=0),
            rhs=lambda v: jnp.concatenate([rhs_a(v), rhs_b(v)], axis=0),
            v_in_lhs=merged_lhs,
            v_in_rhs=merged_rhs,
            slack=merged_slack,
            cst_partition=merged_cst_partition,
            var_partition=merged_var_partition,
            name=name,
            lhs_nz=merged_nz,
        )

    def __add__(self, other: "Constraint") -> "Constraint":
        """Vertically stacks two constraints. See ``add`` for details."""
        return self.add(other)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_shapes(lhs_num: Array, rhs_num: Array) -> None:
        """Validates the dimensions and matching row counts of LHS and RHS arrays.

        Parameters
        ----------
        lhs_num : Array
            The evaluated left-hand side matrix sample.
        rhs_num : Array
            The evaluated right-hand side vector sample.

        Raises
        ------
        ValueError
            If LHS is not 2-D, RHS is not 1-D, or their row counts do not match.
        """
        if lhs_num.ndim != 2:
            raise ValueError(
                f"LHS must return a 2-D array, got ndim={lhs_num.ndim}, "
                f"shape={lhs_num.shape}."
            )
        if rhs_num.ndim != 1:
            raise ValueError(
                f"RHS must return a 1-D array, got ndim={rhs_num.ndim}, "
                f"shape={rhs_num.shape}."
            )
        if lhs_num.shape[0] != rhs_num.shape[0]:
            raise ValueError(
                f"LHS has {lhs_num.shape[0]} rows but RHS has "
                f"{rhs_num.shape[0]} elements."
            )

    def __repr__(self) -> str:
        kind  = "==" if self.is_equality else "<="
        param = "parametric" if self.is_parametric else "constant"
        slack_str = f", slacked={self.slack.n_slack}" if self.is_slacked else ""  # type: ignore
        return (
            f"Constraint({kind}, n_cst={self.n_cst}, n_var={self.n_var}, "
            f"{param}{slack_str})"
        )