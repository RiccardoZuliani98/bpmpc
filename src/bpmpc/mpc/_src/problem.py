"""MPC problem analysis, factorization, and solver attachment.

This module provides the core :class:`MPCProblem` class, which is responsible 
for analyzing the structure of an MPC problem at construction time. The problem 
is defined via objective costs and constraints, which are then decomposed into 
individual mathematical terms. 

To maximize performance under JAX's JIT compilation, the terms are partitioned 
into three categories based on the user-declared ``slow_vars``:
1. **Constant terms**: Evaluated once at build time.
2. **Slow terms**: Evaluated less frequently (e.g., once per episode) via ``prepare``.
3. **Fast terms**: Evaluated every control step via ``solve``.

Constants and slack variables are folded into a base Quadratic Program (QP), 
and the remaining parametric updates are handed to a mode-specific assembler 
('dense' or 'sparse'). 

Finally, a numerical QP solver is attached via :meth:`~MPCProblem.with_solver`, 
producing an optimized :class:`MPCSolver` ready for real-time control.

Usage Example
-------------
.. code-block:: python

    problem = MPCProblem(
        costs=[cost],
        constraints=[dynamics, ic, bounds],
        slow_vars=["p"],
        mode="dense",
    )
    mpc = problem.with_solver(my_qp_solver)

    # Pre-compute slow parameters
    prepared = mpc.prepare({"p": p_val})
    
    # Solve the fast parameters using the prepared QP
    sol = mpc.solve_with_prepared(prepared, {"x0": x0_val})
"""

from __future__ import annotations

from typing import (
    Any, Callable, Dict, List, NamedTuple,
    Optional, Sequence, Tuple, Union, Protocol
)

import warnings
import os

import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO

from bpmpc.mpc._src.assembly import dense, sparse

from bpmpc.variable             import Variable
from bpmpc.mpc._src.cost        import Cost
from bpmpc.mpc._src.constraint  import Constraint
from bpmpc.mpc._src.types       import ArrayIn, QPData, SparseMode
from bpmpc.mpc._src.partition   import Partition
from bpmpc.mpc._src.terms       import (
    Term, decompose_costs, 
    decompose_constraints, 
    collect_variables,
    field_of,
)


# ======================================================================
# MPCSolver
# ======================================================================

class _SolveFn(Protocol):
    def __call__(self, fast_v: ArrayIn, warmstart: Any = None) -> Any: ...
    
class _SolveFnDebug(Protocol):
    def __call__(self, fast_v: ArrayIn, warmstart: Any = None, dump_dir: str | os.PathLike = "") -> Any: ...

class _SolveWithPreparedFn(Protocol):
    def __call__(self, prepared_qp: QPData, fast_v: ArrayIn, warmstart: Any = None) -> Any: ...

class _SolveAllFn(Protocol):
    def __call__(self, v: ArrayIn, warmstart: Any = None) -> Any: ...

class MPCSolver(NamedTuple):
    """Ready-to-call MPC solver produced by attaching a numerical solver to a problem.

    Attributes
    ----------
    prepare : Callable[[ArrayIn], QPData]
        A function that evaluates the "slow" parametric terms and applies them 
        to the constant base QP. Returns a partially updated `QPData` object.
    solve : Callable[[ArrayIn, Any], Any]
        A function that evaluates the "fast" parametric terms, applies them directly 
        to the base QP (ignoring any slow terms), and runs the attached numerical solver.
        Signature is ``solve(fast_v, warmstart=None)``.
    solve_debug : Callable[[ArrayIn, Any], Any]
        Same as solve but it additionally prints the QP matrices to a txt file.
        Make sure you don't jit this function.
    solve_with_prepared : Callable[[QPData, ArrayIn, Any], Any]
        A function that evaluates the "fast" parametric terms, applies them on top 
        of a pre-computed `QPData` (from ``prepare``), and runs the numerical solver.
        Signature is ``solve_with_prepared(prepared_qp, fast_v, warmstart=None)``.
    prepare_all : Callable[[ArrayIn], QPData]
        A function that evaluates both slow and fast parametric terms from a single 
        dictionary and returns the fully assembled `QPData`.
    solve_all : Callable[[ArrayIn, Any], Any]
        A function that assembles the full QP from a single dictionary of slow and 
        fast variables, and runs the numerical solver.
        Signature is ``solve_all(v, warmstart=None)``.
    n_var : int
        The original number of decision variables (excluding slack variables).
    n_dec : int
        The total number of decision variables used by the solver (``n_var + n_slack``).
    n_eq : int
        The total number of equality constraints.
    n_ineq : int
        The total number of inequality constraints (including non-negativity bounds for slacks).
    mode : SparseMode
        The assembly mode used ('dense' or 'sparse').
    all_vars : Dict[str, Variable]
        A dictionary mapping variable names to their corresponding `Variable` descriptors.
    """

    prepare:             Callable[[ArrayIn], QPData]
    solve:               _SolveFn
    solve_debug:         _SolveFnDebug
    solve_with_prepared: _SolveWithPreparedFn
    prepare_all:         Callable[[ArrayIn], QPData]
    solve_all:           _SolveAllFn
    n_var:               int
    n_dec:               int
    n_eq:                int
    n_ineq:              int
    mode:                SparseMode
    all_vars:            Dict[str, Variable]

    def __repr__(self) -> str:
        def _fmt(d: Dict[str, Variable]) -> str:
            if not d:
                return "{}"
            return "{" + ", ".join(f"{k}{v.shape}" for k, v in d.items()) + "}"

        return (
            "MPCSolver("
            f"mode={self.mode!r}, "
            f"n_var={self.n_var}, n_dec={self.n_dec}, "
            f"n_eq={self.n_eq}, n_ineq={self.n_ineq}, "
            f"all_vars={_fmt(self.all_vars)})"
        )


# ======================================================================
# MPCProblem
# ======================================================================

class MPCProblem:
    """Analyzed MPC problem ready for compilation and solver attachment.

    This class parses the user-provided costs and constraints, infers the problem 
    dimensions, allocates slack variables, and builds the underlying assembly 
    mechanism to map dictionaries of parameters into numerical QP arrays.

    Parameters
    ----------
    costs : Sequence[Cost]
        A sequence of `Cost` objects defining the objective function.
    constraints : Sequence[Constraint]
        A sequence of `Constraint` objects defining the problem dynamics and bounds.
    slow_vars : Sequence[str], default ()
        Names of variables that are updated infrequently. These are supplied to 
        the ``prepare`` method. Any variable not listed here is implicitly a 
        *fast* variable supplied to the ``solve`` methods.
    outputs : Optional[Union[Dict[str, PartitionLike], Partition]], default None
        An optional partition map to slice the raw flat primal solution ``x`` into 
        meaningful semantic blocks. Maps a string name to an indexing specification 
        (e.g., ``slice``, ``(start, stop)`` tuple, or an integer array). When provided, 
        the returned ``sol['x']`` becomes a dictionary keyed by these names.
    mode : SparseMode, default "dense"
        The array allocation and assembly mode ('dense' or 'sparse').

    Attributes
    ----------
    n_var : int
        Original number of decision variables.
    n_dec : int
        Total number of decision variables (``n_var + n_slack``).
    n_eq : int
        Number of equality constraints.
    n_ineq : int
        Number of inequality constraints (includes slack variable bounds).
    all_vars : Dict[str, Variable]
        Dictionary of all parametric variables (both slow and fast) required by the problem.
    output_partition : Partition
        Relevant partitions of the primal variables, returned by the solver.
    primal_partition : Partition
        The global structural layout of the primal decision variables (including slacks).
    eq_partition : Partition
        The global structural layout of the equality constraint rows (dual variables).
    ineq_partition : Partition
        The global structural layout of the inequality constraint rows (dual variables, 
        including slack non-negativity bounds).
    """

    def __init__(
        self,
        costs:       Sequence[Cost],
        constraints: Sequence[Constraint],
        slow_vars:   Sequence[str] = (),
        outputs:     Optional[Union[Dict[str, PartitionLike], Partition]] = None,
        mode:        SparseMode    = "dense",
    ) -> None:
        self._mode : SparseMode = mode
        self._slow_vars = frozenset(slow_vars)

        # --- Validate & infer dimensions ---
        if not costs:
            raise ValueError("At least one cost is required.")
        self.n_var = _infer_n_var(costs, constraints)
        
        # ``n_ineq_user`` counts rows coming from user constraints only.
        # ``n_ineq`` (public) additionally includes one non-negativity
        # row per slack variable (``s_i >= 0``).
        self.n_eq, self._n_ineq_user, n_slack = _constraint_dims(constraints)
        self.n_ineq = self._n_ineq_user + n_slack
        self.n_dec  = self.n_var + n_slack

        # --- Decompose and partition ---
        # Flatten all constraints and costs into atomic evaluable pieces (Terms)
        all_terms: List[Term] = [
            *decompose_costs(costs),
            *decompose_constraints(constraints),
        ]
        const_terms, slow_terms, fast_terms = _partition(
            all_terms, self._slow_vars,
        )

        # --- Variable metadata & slow-var sanity check ---
        self.all_vars = collect_variables(slow_terms + fast_terms)
        unknown = self._slow_vars - set(self.all_vars)
        if unknown:
            raise ValueError(
                f"slow_vars contains unknown names {sorted(unknown)}. "
                f"Known variables: {sorted(self.all_vars)}."
            )

        # --- Track which QPData fields can be mutated at solve time ---
        # A field is *fixed* iff no slow or fast term writes to it.
        # Used by the `fixed_elements` property to expose truly-constant
        # pieces of the QP to downstream solvers.
        self._mutable_fields: frozenset[str] = frozenset(
            field_of(t) for t in (*slow_terms, *fast_terms)
        )

        # --- Generate Global Partitions Automatically ---
        self.primal_partition, self.eq_partition, self.ineq_partition, self._registries = \
            _extract_partitions(list(costs), list(constraints), self.n_var)

        # --- Parse Output partition ---
        if isinstance(outputs, Partition):
            self.output_partition = outputs
        else:
            norm_outputs = _normalize_partition(outputs, self.n_dec)
            self.output_partition = Partition(norm_outputs) if norm_outputs is not None else None

        # --- Dispatch to mode-specific assembler ---
        # Assemblers handle how the evaluated Term blocks are physically 
        # injected into the QP arrays (e.g., standard dense scatter vs BCOO updates).
        _builder = sparse.build if mode == "sparse" else dense.build

        self._assembler = _builder(
            const_terms=const_terms,
            slow_terms=slow_terms,
            fast_terms=fast_terms,
            constraints=constraints,
            n_var=self.n_var, 
            n_dec=self.n_dec,
            n_eq=self.n_eq, 
            n_ineq=self.n_ineq,
            n_ineq_user=self._n_ineq_user,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def fixed_elements(self) -> Dict[str, Any]:
        """Extracts QPData fields that no parametric term will ever modify.

        Returns a dictionary containing only the fields of ``base_qp`` that
        are guaranteed to stay constant across all solves — i.e., fields
        with no slow *and* no fast contribution. A field written to by
        any parametric term (slow or fast) is *not* included in the dict.

        In dense mode, the values are standard ``jnp.ndarray``. In sparse mode, 
        matrix fields (``P``, ``A``, ``G``) are ``BCOO`` format arrays.

        Typical use::

            solver = setup_dense_solver(..., fixed_elements=problem.fixed_elements)
            mpc = problem.with_solver(solver)

        Returns
        -------
        Dict[str, Any]
            A dictionary containing unchanging arrays (e.g., ``{"P": P_const, "A": A_const}``).
        """
        base = self._assembler.base_qp
        return {
            name: getattr(base, name)
            for name in ("P", "q", "A", "b", "G", "h")
            if name not in self._mutable_fields
        }
    
    @property
    def sparsity_pattern(self) -> Dict[str, BCOO]:
        """Returns the structural sparsity pattern for sparse mode matrices.

        Returns
        -------
        Dict[str, sparse.BCOO]
            A dictionary containing the sparsity patterns for 'P', 'A', and 'G' 
            as BCOO matrices, where structural non-zeros are marked with 1.0.
            Returns an empty dictionary and raises a warning if in dense mode.
        """
        if self._mode == "dense":
            warnings.warn("Sparsity pattern is not available in 'dense' mode. Returning {}.")
            return {}
            
        base = self._assembler.base_qp
        
        # Reconstruct BCOO matrices filled with 1s to represent the binary pattern
        return {
            "P": BCOO((jnp.ones_like(base.P.data), base.P.indices), shape=base.P.shape), #type: ignore
            "A": BCOO((jnp.ones_like(base.A.data), base.A.indices), shape=base.A.shape), #type: ignore
            "G": BCOO((jnp.ones_like(base.G.data), base.G.indices), shape=base.G.shape), #type: ignore
        }

    def prepare(self, slow_v: ArrayIn) -> QPData:
        """Evaluates and applies the slow parametric terms to the constant base QP.

        Parameters
        ----------
        slow_v : ArrayIn
            A dictionary of evaluated arrays for all variables declared in ``slow_vars``.

        Returns
        -------
        QPData
            A new QPData namedtuple containing the aggregated constant and slow terms.
        """
        a = self._assembler
        return a.apply(a.base_qp, a.slow_terms, slow_v)

    def prepare_all(self, v: ArrayIn) -> QPData:
        """Evaluates and applies both slow and fast parametric terms in one pass.

        Equivalent to ``solve_from(prepare(slow_v), fast_v)``, but takes a single
        dictionary and applies all terms in one ``apply`` call, avoiding the
        intermediate QPData.

        Parameters
        ----------
        v : ArrayIn
            A dictionary of evaluated arrays for all variables (slow and fast).

        Returns
        -------
        QPData
            A completely assembled QPData object ready for numerical resolution.
        """
        a = self._assembler
        return a.apply(a.base_qp, [*a.slow_terms, *a.fast_terms], v)

    def solve_from(self, prepared: QPData, fast_v: ArrayIn) -> QPData:
        """Evaluates and applies the fast parametric terms on top of a prepared QP.

        Parameters
        ----------
        prepared : QPData
            The base or pre-computed QPData object (usually from ``prepare``).
        fast_v : ArrayIn
            A dictionary of evaluated arrays for all fast variables.

        Returns
        -------
        QPData
            A completely assembled QPData object ready for numerical resolution.
        """
        a = self._assembler
        return a.apply(prepared, a.fast_terms, fast_v)

    def with_solver(self, solver: Callable[..., Any]) -> MPCSolver:
        """Attaches a numerical QP solver to the problem configuration.

        Parameters
        ----------
        solver : Callable[..., Any]
            A numerical QP solver function. It must accept arrays ``P, q, A, b, G, h`` 
            as well as a ``warmstart`` argument, and return a dictionary containing 
            at least the primal solution ``"x"``.

        Returns
        -------
        MPCSolver
            A fully configured solver interface exposing ``prepare``, ``solve``, 
            ``solve_with_prepared``, ``prepare_all``, and ``solve_all`` methods.
        """
        partition = self.output_partition

        # --- Shared solve execution & partitioning ---
        def _call_solver(qp: QPData, warmstart: Any) -> Any:
            """Internal helper to execute the solver and partition the output."""
            sol = solver(
                P=qp.P, q=qp.q, A=qp.A, b=qp.b, G=qp.G, h=qp.h,
                warmstart=warmstart,
            )
            if partition is None:
                return sol
                
            # Replace the flat primal array with a dictionary slice view. 
            # Dual variables like ('lam', 'mu') pass through untouched.
            x = sol["x"]
            return {**sol, "x": {name: x[idx] for name, idx in partition.items()}}
        
        counter = 0 # this is a side effect, not very jax-friendly. Make sure not to jit!

        def _call_solver_debug(qp: QPData, warmstart: Any, dump_dir: str | os.PathLike) -> Any:
            """
            Internal helper to execute the solver and partition the output.
            Stores QP ingredients and warmstart in the dump directory specified above.
            """

            sol = solver(
                P=qp.P, q=qp.q, A=qp.A, b=qp.b, G=qp.G, h=qp.h,
                warmstart=warmstart,
            )

            # store qp data
            qp_data = {
                # Using getattr provides extra safety in case the attribute doesn't exist on 'qp' at all
                "P": _safe_to_numpy(getattr(qp, 'P', None)),
                "G": _safe_to_numpy(getattr(qp, 'G', None)),
                "A": _safe_to_numpy(getattr(qp, 'A', None)),
                "q": _safe_to_numpy(getattr(qp, 'q', None)),
                "h": _safe_to_numpy(getattr(qp, 'h', None)),
                "b": _safe_to_numpy(getattr(qp, 'b', None)),
                # Using .get() on the dictionary prevents KeyError if a registry is missing
                "var_names": self._registries.get("var_names", []),
                "eq_names": self._registries.get("eq_names", []),
                "ineq_names": self._registries.get("ineq_names", [])
            }

            # save the dictionary using joblib
            import joblib
            os.makedirs(dump_dir, exist_ok=True)
            file_path = os.path.join(dump_dir, f"qp_data_{counter}.joblib")
            joblib.dump(qp_data, file_path)

            if partition is None:
                return sol
                
            # Replace the flat primal array with a dictionary slice view. 
            # Dual variables like ('lam', 'mu') pass through untouched.
            x = sol["x"]
            return {**sol, "x": {name: x[idx] for name, idx in partition.items()}}

        # --- Public API Closures ---
        def prepare(slow_v: ArrayIn) -> QPData:
            return self.prepare(slow_v)

        def solve(fast_v: ArrayIn, warmstart: Any = None) -> Any:
            """Solves the problem using only the constant base QP and fast variables."""
            qp = self.solve_from(self._assembler.base_qp, fast_v)
            return _call_solver(qp, warmstart)

        def solve_debug(fast_v: ArrayIn, warmstart: Any = None, dump_dir: str | os.PathLike = "") -> Any:
            """Solves the problem using only the constant base QP and fast variables."""
            qp = self.solve_from(self._assembler.base_qp, fast_v)
            return _call_solver_debug(qp, warmstart, dump_dir)

        def solve_with_prepared(
            prepared_qp: QPData, 
            fast_v: ArrayIn, 
            warmstart: Any = None
        ) -> Any:
            """Solves the problem extending a QP initialized via `prepare`."""
            qp = self.solve_from(prepared_qp, fast_v)
            return _call_solver(qp, warmstart)

        def prepare_all(v: ArrayIn) -> QPData:
            return self.prepare_all(v)

        def solve_all(v: ArrayIn, warmstart: Any = None) -> Any:
            """Solves the problem from a single dictionary of slow and fast variables."""
            qp = self.prepare_all(v)
            return _call_solver(qp, warmstart)

        return MPCSolver(
            prepare=prepare, 
            solve=solve, 
            solve_with_prepared=solve_with_prepared,
            prepare_all=prepare_all,
            solve_all=solve_all,
            solve_debug=solve_debug,
            n_var=self.n_var, n_dec=self.n_dec,
            n_eq=self.n_eq, n_ineq=self.n_ineq,
            mode=self._mode, all_vars=self.all_vars,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        def _fmt(d: Dict[str, Variable]) -> str:
            if not d:
                return "{}"
            return "{" + ", ".join(f"{k}{v.shape}" for k, v in d.items()) + "}"

        return (
            "MPCProblem("
            f"mode={self._mode!r}, "
            f"n_var={self.n_var}, n_dec={self.n_dec}, "
            f"n_eq={self.n_eq}, n_ineq={self.n_ineq}, "
            f"slow_vars={sorted(self._slow_vars)}, "
            f"vars={_fmt(self.all_vars)})"
        )


# ======================================================================
# Module-level helpers
# ======================================================================

def _partition(
    terms:     List[Term],
    slow_vars: frozenset[str],
) -> Tuple[List[Term], List[Term], List[Term]]:
    """Splits all problem terms into (constant, slow, fast) lists.

    A term is:
    - **Constant** if it requires no variables.
    - **Slow** if *all* its required variables are present in ``slow_vars``.
    - **Fast** if *any* of its required variables are missing from ``slow_vars``.
    """
    const: List[Term] = []
    slow:  List[Term] = []
    fast:  List[Term] = []
    for t in terms:
        if not t.v_in:
            const.append(t)
        elif frozenset(t.v_in).issubset(slow_vars):
            slow.append(t)
        else:
            fast.append(t)
    return const, slow, fast


def _infer_n_var(
    costs:       Sequence[Cost],
    constraints: Sequence[Constraint],
) -> int:
    """Validates that all objective costs and constraints share the same decision dimension."""
    n_vars = {c.n_var for c in costs} | {c.n_var for c in constraints}
    if len(n_vars) != 1:
        raise ValueError(f"Inconsistent n_var: {n_vars}.")
    return n_vars.pop()


def _constraint_dims(
    constraints: Sequence[Constraint],
) -> Tuple[int, int, int]:
    """Calculates the total row counts for equalities, inequalities, and slack variables."""
    n_eq = n_ineq = total_slack = 0
    for cst in constraints:
        if cst.is_equality:
            n_eq += cst.n_cst
        else:
            n_ineq += cst.n_cst
            if cst.slack is not None:
                total_slack += cst.slack.n_slack
    return n_eq, n_ineq, total_slack


# ======================================================================
# Partition utilities
# ======================================================================

#: Accepted types for a single partition entry. Normalized at build
#: time to either a fast Python ``slice`` or a 1-D int index array.
PartitionLike = Union[slice, Tuple[int, int], Sequence[int], Array]


def _normalize_partition(
    outputs: Optional[Dict[str, PartitionLike]],
    n_dec:   int,
) -> Optional[Dict[str, Union[Array, slice]]]:
    """Converts user-provided partition specifications into normalized formats.

    For performance reasons under JAX's JIT compilation, this function preferentially 
    preserves Python ``slice`` objects. XLA lowers slices to fast, contiguous 
    memory `DynamicSlice` operations. Only non-contiguous lists or explicit arrays 
    are converted to integer index arrays, which XLA lowers to slower memory 
    bandwidth-heavy `Gather` operations.

    Parameters
    ----------
    outputs : Optional[Dict[str, PartitionLike]]
        A dictionary mapping semantic names to output indexing specifications.
    n_dec : int
        The total number of decision variables (used for bounds checking).

    Returns
    -------
    Optional[Dict[str, Union[Array, slice]]]
        A normalized dictionary of slices or integer arrays, or ``None`` if 
        ``outputs`` was ``None``.

    Raises
    ------
    ValueError
        If an array specification is not 1-D, or if any indices (tuples or arrays) 
        fall outside the valid range ``[0, n_dec)``.
    """
    if outputs is None:
        return None

    normalized: Dict[str, Union[Array, slice]] = {}
    for key, spec in outputs.items():
        if isinstance(spec, slice):
            # Resolve start/stop without clamping. A `None` endpoint
            # means "this end of the vector" and is always in range.
            step  = 1 if spec.step is None else int(spec.step)
            if step == 0:
                raise ValueError(f"outputs[{key!r}]: slice step cannot be 0.")
            start = 0              if spec.start is None else int(spec.start)
            stop  = n_dec          if spec.stop  is None else int(spec.stop)

            # Negative indices count from the end
            if start < 0:
                start += n_dec
            if stop  < 0:
                stop  += n_dec

            # Enforce strict bounds (reject instead of clamping)
            if start < 0 or start >= n_dec:
                raise ValueError(
                    f"outputs[{key!r}]: slice indices out of range "
                    f"[0, {n_dec}); got start={spec.start}."
                )
            if stop < 0 or stop > n_dec:
                raise ValueError(
                    f"outputs[{key!r}]: slice indices out of range "
                    f"[0, {n_dec}]; got stop={spec.stop}."
                )
            # Return a fast Python slice object, NOT an array
            normalized[key] = slice(start, stop, step)
            
        else:
            # Fallback to arrays for genuinely non-contiguous indices
            idx = jnp.asarray(spec, dtype=jnp.int32)
            
            # --- Array Validation Logic ---
            if idx.ndim != 1:
                raise ValueError(
                    f"outputs[{key!r}]: indices must be 1-D, got shape {idx.shape}."
                )
                
            # Build-time range check (cheap because indices are concrete at init).
            idx_np = np.asarray(idx)
            if idx_np.size and (idx_np.min() < 0 or idx_np.max() >= n_dec):
                raise ValueError(
                    f"outputs[{key!r}]: indices out of range [0, {n_dec}); "
                    f"got min={int(idx_np.min())}, max={int(idx_np.max())}."
                )
                
            normalized[key] = idx

    return normalized

from typing import Optional

def _process_array_slices(slices_dict: dict[str, slice], array_length: Optional[int] = None):
    """
    Finds the partition of the array with the most overlapping slices, 
    and returns a padded/trimmed representation of the array variables,
    prioritizing the shortest slice when overlaps occur.
    
    Parameters
    ----------
    slices_dict: A dictionary mapping variable names to their slices.
    array_length: The total length of the array. If None, it is inferred 
                  from the maximum 'stop' value in the slices.
    """
    # 1. Infer array length if not provided
    if array_length is None:
        max_stop = 0
        for s in slices_dict.values():
            if s.stop is not None:
                max_stop = max(max_stop, s.stop)
        array_length = max_stop

    if array_length == 0:
        return [], []

    # 2. Find the partition (indices) with the most slices
    overlap_counts = [0] * array_length
    for s in slices_dict.values():
        for i in range(*s.indices(array_length)):
            overlap_counts[i] += 1
            
    max_overlap = max(overlap_counts) if overlap_counts else 0
    max_overlap_indices = [i for i, count in enumerate(overlap_counts) if count == max_overlap]

    # 3. Trim (unroll) variables and keep only the shortest slice per index
    # Store tuples of (slice_length, entry_name) or None
    named_array = [None] * array_length
    
    # Process known variables
    for var_name, s in slices_dict.items():
        indices = range(*s.indices(array_length))
        slice_length = len(indices)
        is_multiple = slice_length > 1
        
        for count, idx in enumerate(indices, start=1):
            entry_name = f"{var_name}_{count}" if is_multiple else var_name
            
            # Only assign if the index is empty, or if this new slice is strictly shorter
            if named_array[idx] is None or slice_length < named_array[idx][0]:
                named_array[idx] = (slice_length, entry_name)

    # 4. Pad unknowns and finalize the array
    final_array = []
    unknown_counter = 1
    
    for i in range(array_length):
        if named_array[i] is None:
            # Pad empty spots with unknowns
            final_array.append(f"unknown_{unknown_counter}")
            unknown_counter += 1
        else:
            # Extract just the entry_name from our tuple
            final_array.append(named_array[i][1])

    return max_overlap_indices, final_array

def _extract_partitions(
    costs: List[Cost], 
    constraints: List[Constraint], 
    n_var: int
) -> Tuple[Partition, Partition, Partition, Dict[str, List[str]]]:
    """Generates the global metadata partitions for the MPC problem.
    
    This analyzes the individual `var_partition` and `cst_partition` metadata 
    attached to costs and constraints, merging them into global layouts. 
    It automatically handles the appended decision variables and non-negativity 
    bounds required when slack variables are introduced.

    Parameters
    ----------
    costs: List of objective costs.
    constraints: List of problem constraints.
    n_var: The base number of decision variables (excluding slacks).

    Returns
    -------
    A tuple of three `Partition` objects:
      - `primal_partition`: Global layout of the decision vector (including slacks).
      - `eq_partition`: Global layout of the equality constraint rows.
      - `ineq_partition`: Global layout of the inequality constraint rows (including slack bounds).
    registry: A dictionary containing the parsed variable arrays for 'var_names', 
        'eq_names', and 'ineq_names'.
    """
    eq_mapping: Dict[str, Union[slice, Array, dict]] = {}
    ineq_mapping: Dict[str, Union[slice, Array, dict]] = {}
    primal_mapping: Dict[str, Union[slice, Array]] = {}

    # Cache for detailed slack names to avoid duplicate processing in Step 3
    slack_names_cache: Dict[str, List[str]] = {}

    # 1. Scrape and Validate the base Primal Partition
    base_var_partition = None
    for item in costs + constraints:
        vp = getattr(item, "var_partition", None)
        if vp is not None:
            if base_var_partition is None:
                base_var_partition = vp
            else:
                for k, v in vp._mapping.items():
                    if k in base_var_partition._mapping:
                        v_base = base_var_partition._mapping[k]
                        if isinstance(v, slice) and isinstance(v_base, slice):
                            if (v.start, v.stop, v.step) != (v_base.start, v_base.stop, v_base.step):
                                raise ValueError(f"var_partition conflict on key '{k}'.")
                        elif not isinstance(v, slice) and not isinstance(v_base, slice):
                            if not np.array_equal(np.asarray(v), np.asarray(v_base)):
                                raise ValueError(f"var_partition array conflict on key '{k}'.")
                        else:
                            raise ValueError(f"var_partition type mismatch on key '{k}'.")

    if base_var_partition is not None:
        primal_mapping.update(base_var_partition._mapping)

    # Counters for global row/column tracking
    n_eq = 0
    n_ineq_user = 0
    total_slack = 0

    # 2. Process Equality and User Inequality constraints
    for i, cst in enumerate(constraints):
        
        name = getattr(cst,"name",f"cst_{i}")

        if cst.is_equality:
            if cst.cst_partition is not None:
                shifted_partition = cst.cst_partition.offset(n_eq)
                # Flatten the keys into the global mapping (prefixed to avoid collisions)
                for k, v in shifted_partition._mapping.items():
                    eq_mapping[k] = v
            else:
                eq_mapping[name] = slice(n_eq, n_eq + cst.n_cst)
            n_eq += cst.n_cst
            
        else: # is inequality
            if cst.cst_partition is not None:
                shifted_partition = cst.cst_partition.offset(n_ineq_user)
                # Flatten the keys into the global mapping
                for k, v in shifted_partition._mapping.items():
                    ineq_mapping[k] = v
            else:
                ineq_mapping[name] = slice(n_ineq_user, n_ineq_user + cst.n_cst)
            n_ineq_user += cst.n_cst
            
            # Handle Slack Primal Columns
            if cst.slack is not None:
                # Fallback to an empty dict if the constraint doesn't have a specific partition mapped
                cst_mapping = cst.cst_partition._mapping if cst.cst_partition is not None else {}
                
                # Fetch detailed names using imported function
                _, parsed_array = _process_array_slices(cst_mapping, array_length=cst.slack.n_slack)
                slack_names_cache[name] = parsed_array
                
                # Map each individual slack entry to a 1D slice
                for offset, s_detail in enumerate(parsed_array):
                    s_name = f"slack_{name}_{s_detail}"
                    primal_mapping[s_name] = slice(
                        n_var + total_slack + offset, 
                        n_var + total_slack + offset + 1
                    )
                    
                total_slack += cst.slack.n_slack

    # 3. Process Slack Non-Negativity Bounds (Appended to Inequalities)
    current_slack_ineq_row = n_ineq_user
    for i, cst in enumerate(constraints):
        if not cst.is_equality and cst.slack is not None:
            name = getattr(cst, 'name', f"cst_{i}") 
            parsed_array = slack_names_cache[name]
            
            # Map each individual slack non-negativity bound to a 1D slice
            for offset, s_detail in enumerate(parsed_array):
                slack_bound_name = f"slack_nonneg_{name}_{s_detail}"
                ineq_mapping[slack_bound_name] = slice(
                    current_slack_ineq_row + offset, 
                    current_slack_ineq_row + offset + 1
                )
                
            current_slack_ineq_row += cst.slack.n_slack

    # 4. Generate the Registry
    # Extract only the parsed variables (index 1 of the returned tuple) from the process_array_slices function
    registries = {
        "var_names": _process_array_slices(
            {k: v for k, v in primal_mapping.items() if isinstance(v, slice)}, 
            array_length=n_var + total_slack
        )[1],
        "eq_names": _process_array_slices(
            {k: v for k, v in eq_mapping.items() if isinstance(v, slice)}, 
            array_length=n_eq
        )[1],
        "ineq_names": _process_array_slices(
            {k: v for k, v in ineq_mapping.items() if isinstance(v, slice)}, 
            array_length=current_slack_ineq_row
        )[1],
    }

    return Partition(primal_mapping), Partition(eq_mapping), Partition(ineq_mapping), registries


# ======================================================================
# Numpy conversion utils
# ======================================================================

def _safe_to_numpy(mat):
    if mat is None:
        return None
    if hasattr(mat, 'toarray'):
        # SciPy sparse
        return np.asarray(mat.toarray())
    elif hasattr(mat, 'todense'):
        # JAX BCOO or older SciPy sparse
        return np.asarray(mat.todense())
    else:
        # Standard JAX arrays, NumPy arrays, or lists
        return np.asarray(mat)