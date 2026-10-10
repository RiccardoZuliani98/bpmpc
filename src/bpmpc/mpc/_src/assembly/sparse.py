"""Sparse QP assembly via coordinate lists and XLA-fused BCOO indexing.

This module implements the sparse equivalent to the dense QP assembler.
Rather than allocating full dense matrices for the QP solver, it collects
the structural non-zeros of the MPC problem at build time.

Every matrix term contributes a :class:`~bpmpc.mpc._src.types.NonZeros`:
static coordinates plus a callable producing the values at them.  Terms
built by the helpers declare theirs, so their dense matrix is never built.
For any other term the pattern is found by **random probing**: evaluating
the dense callable with dummy random inputs and keeping the entries that
are non-zero.  Slack coefficients are constant coordinate lists as well.

The coordinates are appended, in a fixed order, into one static
`jax.experimental.sparse.BCOO` pattern per matrix, each term owning a
contiguous slice of its data.  At runtime (inside JIT), the `apply`
method overwrites those slices without rebuilding the sparsity graph.
"""

from typing import List, Dict, Union, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from collections import defaultdict

from bpmpc.mpc._src.types import QPData, NonZeros
from bpmpc.mpc._src.terms import field_of, decompose_slacks, _CostTerm, _CstTerm
from bpmpc.mpc._src.constraint import Constraint

class SparseAssembler:
    """Assembles QP matrices using static BCOO updates.

    This class statically determines the sparsity pattern of the entire MPC 
    problem during instantiation. At runtime, its `apply` method functionally 
    mutates the sparse arrays, enabling extremely fast, XLA-optimized updates.

    Attributes
    ----------
    shapes : dict
        The global dense shapes of the QP fields (P, q, A, b, G, h, c).
    matrix_metadata : dict
        Stores the static BCOO `indices`, the 1D data sizes, and for each
        parametric term the `slice` of the data it owns and its value callable.
    base_qp : QPData
        The statically compiled QP structure containing all constant terms, 
        slack penalties, and the static BCOO sparsity indices.
    """
    
    def __init__(
        self, 
        const_terms: List[Union[_CostTerm, _CstTerm]], 
        slow_terms: List[Union[_CostTerm, _CstTerm]], 
        fast_terms: List[Union[_CostTerm, _CstTerm]], 
        constraints: Sequence[Constraint], 
        n_var: int, 
        n_dec: int, 
        n_eq: int, 
        n_ineq: int, 
        n_ineq_user: int
    ) -> None:
        """Initializes the SparseAssembler and bakes the static structures."""
        self.slow_terms = slow_terms
        self.fast_terms = fast_terms
        
        self.shapes = {
            'P': (n_dec, n_dec), 'q': (n_dec,),
            'A': (n_eq, n_dec),  'b': (n_eq,),
            'G': (n_ineq, n_dec),'h': (n_ineq,),
            'c': ()
        }
        
        # Group all terms by their target field (P, q, A, etc.) and update speed.
        self.terms_by_field = {f: {'const': [], 'slow': [], 'fast': []} for f in self.shapes}
        for t in const_terms: self.terms_by_field[field_of(t)]['const'].append(t)
        for t in slow_terms:  self.terms_by_field[field_of(t)]['slow'].append(t)
        for t in fast_terms:  self.terms_by_field[field_of(t)]['fast'].append(t)
        
        # Slacks only contribute constant coordinate lists to P, q and G.
        slack_P, slack_G, slack_q = decompose_slacks(constraints, n_var, n_ineq_user)
        slack_coo = {'P': slack_P, 'G': slack_G}

        # ------------------------------------------------------------------
        # Build Matrix Structures (P, A, G)
        # ------------------------------------------------------------------
        self.matrix_metadata = {'P': {}, 'A': {}, 'G': {}}
        base_qp_kwargs = {}
        
        for field in ['P', 'A', 'G']:
            # Each entry appends global coordinates and their constant values
            # (zeros for parametric terms, filled in by apply()).
            all_rows, all_cols, all_data = [], [], []
            current_idx = 0
            term_slices = {}
            
            # 1. Process terms in a strictly deterministic order, reserving a
            # contiguous slice of the BCOO data array for each term.
            for category in ['const', 'slow', 'fast']:
                for term in self.terms_by_field[field][category]:
                    nz = term.nz if term.nz is not None else probe_term(term)
                    nnz = len(nz.rows)
                    
                    term_slices[term.uid] = {
                        'slice': slice(current_idx, current_idx + nnz),
                        'vals': nz.vals,
                    }
                    all_rows.append(nz.rows + _row_offset(term))
                    all_cols.append(nz.cols)
                    all_data.append(
                        np.asarray(nz.vals({})) if category == 'const' else np.zeros(nnz)
                    )
                    current_idx += nnz
            
            # 2. Append the constant slack coefficients at the end of the array.
            if field in slack_coo:
                rows, cols, vals = slack_coo[field]
                all_rows.append(rows)
                all_cols.append(cols)
                all_data.append(vals)
                current_idx += len(rows)

            # 3. Concatenate all indices into a single (Total_NNZ, 2) XLA-compatible array.
            if all_rows:
                global_r = np.concatenate(all_rows)
                global_c = np.concatenate(all_cols)
                indices = jnp.array(np.stack([global_r, global_c], axis=1), dtype=jnp.int32)
                data = np.concatenate(all_data).astype(np.float64)
            else:
                indices = jnp.empty((0, 2), dtype=jnp.int32)
                data = np.zeros(0, dtype=np.float64)
                
            self.matrix_metadata[field] = {
                'indices': indices,
                'term_slices': term_slices,
                'total_nnz': current_idx
            }

            # Create the frozen BCOO construct.
            base_qp_kwargs[field] = sparse.BCOO(
                (jnp.array(data), indices), 
                shape=self.shapes[field]
            )
            
        # ------------------------------------------------------------------
        # Build Vector Structures (q, b, h, c)
        # ------------------------------------------------------------------
        # Vector structures remain dense. We evaluate constants once and 
        # position them in the correct slices of the global vectors.
        for field in ['q', 'b', 'h', 'c']:
            val = np.zeros(self.shapes[field], dtype=np.float64)
            for term in self.terms_by_field[field]['const']:
                out = np.array(term.fn({}))
                
                if isinstance(term, _CostTerm):
                    if field == 'c':
                        val += out
                    else:
                        val[:out.shape[0]] += out
                else:
                    val[term.row_start : term.row_end] += out
                    
            # Burn slack linear penalties into the bottom block of q.
            if field == 'q':
                val[n_var : n_dec] += slack_q
                
            base_qp_kwargs[field] = jnp.array(val)
            
        self.base_qp = QPData(**base_qp_kwargs)

    def apply(
        self, 
        qp: QPData, 
        terms: List[Union[_CostTerm, _CstTerm]], 
        vars: Dict[str, jax.Array]
    ) -> QPData:
        """Evaluates parametric terms and updates the QP arrays.

        Designed to be compiled by `jax.jit`. It iterates over the specified 
        list of terms (either slow or fast), evaluates them, and functionally 
        mutates the underlying `BCOO.data` or dense vector arrays.

        Because sparsity patterns are fixed and pre-calculated, XLA will 
        unroll these loops and fuse the slice updates into highly efficient 
        memory writes.

        Parameters
        ----------
        qp : QPData
            The base QP state to update.
        terms : list of _CostTerm or _CstTerm
            The parametric terms to evaluate (e.g., `self.fast_terms`).
        vars : dict of str to Array
            The runtime numerical values of the variables required by the terms.

        Returns
        -------
        QPData
            A new `QPData` object containing the updated constraints and costs.
        """
        updates = {field: getattr(qp, field) for field in self.shapes}
            
        terms_by_field = defaultdict(list)
        for t in terms:
            terms_by_field[field_of(t)].append(t)
            
        # 1. Update Matrix Fields (BCOO mutations)
        for field in ['P', 'A', 'G']:
            if field not in terms_by_field:
                continue
                
            meta = self.matrix_metadata[field]
            new_data = updates[field].data
            
            for term in terms_by_field[field]:
                t_meta = meta['term_slices'][term.uid]
                # Overwrite this term's dedicated 1D slice in the BCOO data array.
                new_data = new_data.at[t_meta['slice']].set(t_meta['vals'](vars))
                
            # Re-package the updated 1D data with the original static indices.
            updates[field] = sparse.BCOO(
                (new_data, meta['indices']), 
                shape=self.shapes[field]
            )
            
        # 2. Update Vector Fields (Dense mutations)
        for field in ['q', 'b', 'h', 'c']:
            if field not in terms_by_field:
                continue
                
            new_val = updates[field]
            for term in terms_by_field[field]:
                out = term.fn(vars)
                
                if isinstance(term, _CostTerm):
                    if field == 'c':
                        new_val += out
                    else:
                        new_val = new_val.at[:out.shape[0]].add(out)
                else:
                    new_val = new_val.at[term.row_start : term.row_end].add(out)
                
            updates[field] = new_val
            
        return QPData(**updates)


def build(
    const_terms: List[Union[_CostTerm, _CstTerm]], 
    slow_terms: List[Union[_CostTerm, _CstTerm]], 
    fast_terms: List[Union[_CostTerm, _CstTerm]], 
    constraints: Sequence[Constraint], 
    n_var: int, 
    n_dec: int, 
    n_eq: int, 
    n_ineq: int, 
    n_ineq_user: int
) -> SparseAssembler:
    """Factory matching the expected signature in `problem.py`.

    Instantiates and returns the `SparseAssembler` object which manages 
    the static problem state and dynamically updates it during simulation.
    """
    return SparseAssembler(
        const_terms=const_terms, 
        slow_terms=slow_terms, 
        fast_terms=fast_terms, 
        constraints=constraints,
        n_var=n_var,
        n_dec=n_dec, 
        n_eq=n_eq, 
        n_ineq=n_ineq, 
        n_ineq_user=n_ineq_user,
    )

# ======================================================================
# Internal
# ======================================================================

def _row_offset(term: Union[_CostTerm, _CstTerm]) -> int:
    """Global row of a term's local row 0.

    Cost terms sit in the top-left ``[n_var, n_var]`` block of P; constraint
    terms are shifted down to their slice of the global constraint matrix.
    """
    return term.row_start if isinstance(term, _CstTerm) else 0


def probe_term(
    term: Union[_CostTerm, _CstTerm], 
    num_probes: int = 3, 
    key_seed: int = 42, 
    tol: float = 1e-10
) -> NonZeros:
    """Evaluates a matrix term with random inputs to find structural non-zeros.

    Fallback for terms that do not declare their non-zeros. By evaluating
    the term multiple times with normally distributed random inputs, we can
    reliably identify structurally non-zero elements while avoiding
    "accidental zeros" that might occur with a single evaluation.
    This must be run statically at build time (outside of JIT).

    Parameters
    ----------
    term : _CostTerm or _CstTerm
        The matrix term to probe.
    num_probes : int, optional
        Number of random evaluations to perform (default is 3).
    key_seed : int, optional
        Random seed for JAX PRNG (default is 42).
    tol : float, optional
        Tolerance below which an element is considered structurally zero.

    Returns
    -------
    NonZeros
        The term-local coordinates of the non-zeros, with a value callable
        that gathers them from the dense output of ``term.fn``.
    """
    key = jax.random.PRNGKey(key_seed)
    accum_mask = None
    
    for _ in range(num_probes):
        dummy_inputs = {}
        if term.v_in:
            for name, var in term.v_in.items():
                key, subkey = jax.random.split(key)
                dummy_inputs[name] = jax.random.normal(subkey, var.shape)
            
        out = term.fn(dummy_inputs)
        current_mask = jnp.abs(out) > tol
        
        if accum_mask is None:
            accum_mask = current_mask
        else:
            accum_mask = accum_mask | current_mask
            
    rows, cols = np.nonzero(np.array(accum_mask))
    fn = term.fn
    return NonZeros(rows, cols, lambda v: fn(v)[rows, cols])
