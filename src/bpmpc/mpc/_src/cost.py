"""Cost descriptor for parametric MPC problems.

A cost has the form:

    0.5 * z^T Q(v) z  +  q(v)^T z  +  c(v)

Each of the three terms can depend on a different set of variables.
``v_in_*=None`` means that term is constant (the callable receives
``{}``). Omitting ``q_vec`` or ``c`` entirely means the term is zero.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import jax.numpy as jnp
from jax import Array

from bpmpc.variable             import Variable
from bpmpc.mpc._src.types       import ArrayIn
from bpmpc.mpc._src.validation  import validate_shared_variables, make_sample, merge_v_in
from bpmpc.mpc._src.partition   import Partition


class Cost:
    """Parametric quadratic cost ``0.5 z^T Q(v) z + q(v)^T z + c(v)``.

    Parameters
    ----------
    q_mat : Union[Array, Callable[[ArrayIn], Array]]
        Callable returning the symmetric quadratic matrix term, shape ``(n_var, n_var)``, or a constant array.
    q_vec : Optional[Union[Array, Callable[[ArrayIn], Array]]], default None
        Callable returning the linear vector term, shape ``(n_var,)``, or a constant array.
    c : Optional[Union[Array, float, Callable[[ArrayIn], Array]]], default None
        Callable returning the scalar offset term, or a constant scalar.
    v_in_q_mat : Optional[Dict[str, Variable]], default None
        Dictionary of variables required by the ``q_mat`` callable, or ``None`` if constant.
    v_in_q_vec : Optional[Dict[str, Variable]], default None
        Dictionary of variables required by the ``q_vec`` callable, or ``None`` if constant.
    v_in_c : Optional[Dict[str, Variable]], default None
        Dictionary of variables required by the ``c`` callable, or ``None`` if constant.
    var_partition : Optional[Partition], default None
        Description of each variable using the "Partition" class.

    Attributes
    ----------
    v_in_q_mat : Optional[Dict[str, Variable]]
        Variables required by the quadratic term.
    v_in_q_vec : Optional[Dict[str, Variable]]
        Variables required by the linear term.
    v_in_c : Optional[Dict[str, Variable]]
        Variables required by the scalar offset term.
    n_var : int
        The dimension of the decision variable vector ``z``.
    var_partition : Optional[Partition]
        Description of each variable using the "Partition" class.

    Raises
    ------
    ValueError
        If ``q_mat`` does not return a 2-D square array.
        If ``q_vec`` does not return a 1-D array matching ``n_var``.
        If ``c`` does not return a scalar.
    """

    v_in_q_mat:     Optional[Dict[str, Variable]]
    v_in_q_vec:     Optional[Dict[str, Variable]]
    v_in_c:         Optional[Dict[str, Variable]]
    n_var:          int
    var_partition:  Optional[Partition]

    def __init__(
        self,
        q_mat:          Union[Array, Callable[[ArrayIn], Array]],
        q_vec:          Optional[Union[Array, Callable[[ArrayIn], Array]]] = None,
        c:              Optional[Union[Array, float, Callable[[ArrayIn], Array]]] = None,
        v_in_q_mat:     Optional[Dict[str, Variable]] = None,
        v_in_q_vec:     Optional[Dict[str, Variable]] = None,
        v_in_c:         Optional[Dict[str, Variable]] = None,
        var_partition:  Optional[Partition] = None,
    ) -> None:
        self.v_in_q_mat = v_in_q_mat
        self.v_in_q_vec = v_in_q_vec
        self.v_in_c     = v_in_c

        # Cross-check shared variables between terms.
        if v_in_q_mat is not None and v_in_q_vec is not None:
            validate_shared_variables(v_in_q_mat, v_in_q_vec, "q_mat", "q_vec")
        if v_in_q_mat is not None and v_in_c is not None:
            validate_shared_variables(v_in_q_mat, v_in_c, "q_mat", "c")
        if v_in_q_vec is not None and v_in_c is not None:
            validate_shared_variables(v_in_q_vec, v_in_c, "q_vec", "c")

        # --------------------------------------------------------------
        # Normalize and Probe q_mat
        # --------------------------------------------------------------
        if callable(q_mat):
            _q_mat_fn = q_mat
        else:
            _q_mat_arr = jnp.asarray(q_mat)
            _q_mat_fn = lambda _: _q_mat_arr

        q_mat_num = _q_mat_fn(make_sample(v_in_q_mat))
        if q_mat_num.ndim != 2:
            raise ValueError(
                f"q_mat must return a 2-D array, got ndim={q_mat_num.ndim}, "
                f"shape={q_mat_num.shape}."
            )
        n_row, n_col = q_mat_num.shape
        if n_row != n_col:
            raise ValueError(f"q_mat must be square, got shape ({n_row}, {n_col}).")
        
        self.n_var = n_row
        self._q_mat = _q_mat_fn

        self.var_partition = var_partition

        # --------------------------------------------------------------
        # Normalize and Probe q_vec
        # --------------------------------------------------------------
        if q_vec is not None:
            if callable(q_vec):
                _q_vec_fn = q_vec
            else:
                _q_vec_arr = jnp.asarray(q_vec)
                _q_vec_fn = lambda _: _q_vec_arr

            q_vec_num = _q_vec_fn(make_sample(v_in_q_vec))
            if q_vec_num.ndim != 1:
                raise ValueError(
                    f"q_vec must return a 1-D array, got ndim={q_vec_num.ndim}, "
                    f"shape={q_vec_num.shape}."
                )
            if q_vec_num.shape[0] != self.n_var:
                raise ValueError(
                    f"q_vec has length {q_vec_num.shape[0]} but q_mat is "
                    f"{self.n_var}x{self.n_var}."
                )
            self._q_vec = _q_vec_fn
        else:
            n = self.n_var
            self._q_vec = lambda _: jnp.zeros((n,))

        # --------------------------------------------------------------
        # Normalize and Probe c
        # --------------------------------------------------------------
        if c is not None:
            if callable(c):
                _c_fn = c
            else:
                _c_arr = jnp.asarray(c)
                _c_fn = lambda _: _c_arr

            c_num = _c_fn(make_sample(v_in_c))
            if c_num.ndim != 0:
                raise ValueError(
                    f"c must return a scalar, got ndim={c_num.ndim}, "
                    f"shape={c_num.shape}."
                )
            self._c = _c_fn
        else:
            self._c = lambda _: jnp.array(0.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_parametric_q_mat(self) -> bool:
        """bool: True if the quadratic term depends on runtime variables."""
        return self.v_in_q_mat is not None and len(self.v_in_q_mat) > 0

    @property
    def is_parametric_q_vec(self) -> bool:
        """bool: True if the linear term depends on runtime variables."""
        return self.v_in_q_vec is not None and len(self.v_in_q_vec) > 0

    @property
    def is_parametric_c(self) -> bool:
        """bool: True if the scalar term depends on runtime variables."""
        return self.v_in_c is not None and len(self.v_in_c) > 0

    @property
    def is_parametric(self) -> bool:
        """bool: True if any term in the cost relies on runtime variables."""
        return (self.is_parametric_q_mat
                or self.is_parametric_q_vec
                or self.is_parametric_c)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def eval_q_mat(self, v: Optional[ArrayIn] = None) -> Array:
        """Evaluates the quadratic matrix component.

        Parameters
        ----------
        v : Optional[ArrayIn], default None
            Dictionary of runtime array inputs.

        Returns
        -------
        Array
            The evaluated quadratic matrix of shape ``(n_var, n_var)``.
        """
        return self._q_mat(v if v is not None else {})

    def eval_q_vec(self, v: Optional[ArrayIn] = None) -> Array:
        """Evaluates the linear vector component.

        Parameters
        ----------
        v : Optional[ArrayIn], default None
            Dictionary of runtime array inputs.

        Returns
        -------
        Array
            The evaluated linear vector of shape ``(n_var,)``.
        """
        return self._q_vec(v if v is not None else {})

    def eval_c(self, v: Optional[ArrayIn] = None) -> Array:
        """Evaluates the scalar offset component.

        Parameters
        ----------
        v : Optional[ArrayIn], default None
            Dictionary of runtime array inputs.

        Returns
        -------
        Array
            The evaluated scalar offset.
        """
        return self._c(v if v is not None else {})

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------
    def add(self, other: "Cost") -> "Cost":
        """Sums two costs element-wise.

        Parameters
        ----------
        other : Cost
            Another Cost object to be added to this one.

        Returns
        -------
        Cost
            A newly constructed Cost object representing the sum.

        Raises
        ------
        ValueError
            If the dimensions (``n_var``) of the two costs do not match.
        """
        if self.n_var != other.n_var:
            raise ValueError(
                f"Cannot add costs with different n_var: "
                f"{self.n_var} vs {other.n_var}."
            )

        merged_q_mat = merge_v_in(self.v_in_q_mat, other.v_in_q_mat,
                                  "self.q_mat", "other.q_mat")
        merged_q_vec = merge_v_in(self.v_in_q_vec, other.v_in_q_vec,
                                  "self.q_vec", "other.q_vec")
        merged_c     = merge_v_in(self.v_in_c, other.v_in_c,
                                  "self.c", "other.c")

        qm_a, qm_b = self._q_mat, other._q_mat
        qv_a, qv_b = self._q_vec, other._q_vec
        c_a,  c_b  = self._c,     other._c

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
            mergedvar_partition = other.var_partition

        return Cost(
            q_mat=lambda v: qm_a(v) + qm_b(v),
            q_vec=lambda v: qv_a(v) + qv_b(v),
            c    =lambda v: c_a(v)  + c_b(v),
            v_in_q_mat=merged_q_mat,
            v_in_q_vec=merged_q_vec,
            v_in_c=merged_c,
            var_partition=merged_var_partition
        )

    def __add__(self, other: "Cost") -> "Cost":
        """Sums two costs element-wise. See ``add`` for details."""
        return self.add(other)

    def __repr__(self) -> str:
        param = "parametric" if self.is_parametric else "constant"
        return f"Cost(n_var={self.n_var}, {param})"