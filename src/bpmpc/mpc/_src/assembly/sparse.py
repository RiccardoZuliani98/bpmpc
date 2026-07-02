"""Sparse QP assembly via random probing and XLA-fused BCOO indexing.

This module implements the sparse equivalent to the dense QP assembler.
Rather than allocating full dense matrices for the QP solver, it analyzes 
the structural sparsity of the MPC problem at build time. 

It accomplishes this via **random probing**: evaluating the user-defined 
cost and constraint functions with dummy random inputs to discover which 
matrix entries are non-zero. It then compiles a static `jax.experimental.sparse.BCOO` 
sparsity pattern. At runtime (inside JIT), the `apply` method rapidly 
updates these specific non-zero slices without rebuilding the sparsity graph.
"""

from typing import List, Dict, Tuple, Union, Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from collections import defaultdict

from bpmpc.mpc._src.types import QPData
from bpmpc.mpc._src.terms import field_of, _CostTerm, _CstTerm
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
        Stores the static BCOO `indices`, the 1D data sizes, and the exact 
        slice objects that dictate where each parametric term writes its non-zeros.
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
        
        # ------------------------------------------------------------------
        # Extract Slack Structure
        # ------------------------------------------------------------------
        # Slacks only contribute linear/quadratic costs and constant -1.0 
        # couplings in the G matrix. They are perfectly static.
        n_slack = n_dec - n_var
        
        # slack_P_rows_old, slack_P_cols_old, slack_P_vals_old = [], [], []
        # slack_G_rows_old, slack_G_cols_old, slack_G_vals_old = [], [], []
        # q_slack_vec_old = np.zeros(n_slack)
        
        # if n_slack > 0:
        #     col = row = 0
        #     for cst in constraints:
        #         if cst.is_equality: 
        #             continue
        #         if cst.slack and not cst.slack.is_empty:
        #             s = cst.slack
        #             # Extract Diagonal P_slack and dense q_slack
        #             for i in range(s.n_slack):
        #                 slack_P_rows_old.append(n_var + col + i)
        #                 slack_P_cols_old.append(n_var + col + i)
        #                 slack_P_vals_old.append(float(s.w_quad_array[i]))
        #                 q_slack_vec_old[col + i] = float(s.w_lin_array[i])
                        
        #             # Extract G_slack coupling (-1.0 maps slack to inequalities)
        #             for i, idx in enumerate(s.slack_indices):
        #                 slack_G_rows_old.append(row + idx)
        #                 slack_G_cols_old.append(n_var + col + i)
        #                 slack_G_vals_old.append(-1.0)
        #             col += s.n_slack
        #         row += cst.n_cst
                
        #     # Extract non-negativity bounds for all slack variables (-s <= 0)
        #     for i in range(n_slack):
        #         slack_G_rows_old.append(n_ineq_user + i)
        #         slack_G_cols_old.append(n_var + i)
        #         slack_G_vals_old.append(-1.0)

        slack_P_rows, slack_P_cols, slack_P_vals = [], [], []
        slack_G_rows, slack_G_cols, slack_G_vals = [], [], []
        q_slack_vec = np.zeros(n_slack)

        if n_slack > 0:
            col = row = 0
            for cst in constraints:
                if cst.is_equality: 
                    continue
                
                if cst.slack and not cst.slack.is_empty:
                    s = cst.slack
                    n_s = s.n_slack
                    
                    # 1. Extract Diagonal P_slack and dense q_slack
                    # Use range() directly to extend lists in bulk
                    idx_range = range(n_var + col, n_var + col + n_s)
                    slack_P_rows.extend(idx_range)
                    slack_P_cols.extend(idx_range)

                    # multiply by two to counteract the 0.5 imposed by the QP solver
                    slack_P_vals.extend(2.0 * float(x) for x in s.w_quad_array[:n_s])
                    
                    # Use slice assignment for bulk updating the q_slack_vec
                    q_slack_vec[col : col + n_s] = [float(x) for x in s.w_lin_array[:n_s]]
                        
                    # 2. Extract G_slack coupling (-1.0 maps slack to inequalities)
                    num_idx = len(s.slack_indices)
                    slack_G_rows.extend(row + idx for idx in s.slack_indices)
                    slack_G_cols.extend(range(n_var + col, n_var + col + num_idx))
                    
                    # Multiply a single element list to bulk-create constant values
                    slack_G_vals.extend([-1.0] * num_idx)
                    
                    col += n_s
                
                row += cst.n_cst
                
            # 3. Extract non-negativity bounds for all slack variables (-s <= 0)
            slack_G_rows.extend(range(n_ineq_user, n_ineq_user + n_slack))
            slack_G_cols.extend(range(n_var, n_var + n_slack))
            slack_G_vals.extend([-1.0] * n_slack)

        # --- ASSERTIONS ---
        # assert slack_P_rows == slack_P_rows_old, "Mismatch in slack_P_rows"
        # assert slack_P_cols == slack_P_cols_old, "Mismatch in slack_P_cols"
        # assert slack_P_vals == slack_P_vals_old, "Mismatch in slack_P_vals"
        
        # assert slack_G_rows == slack_G_rows_old, "Mismatch in slack_G_rows"
        # assert slack_G_cols == slack_G_cols_old, "Mismatch in slack_G_cols"
        # assert slack_G_vals == slack_G_vals_old, "Mismatch in slack_G_vals"
        
        # assert np.allclose(q_slack_vec, q_slack_vec_old), "Mismatch in q_slack_vec"

        # ------------------------------------------------------------------
        # Build Matrix Structures (P, A, G)
        # ------------------------------------------------------------------
        self.matrix_metadata = {'P': {}, 'A': {}, 'G': {}}
        base_qp_kwargs = {}
        
        for field in ['P', 'A', 'G']:
            all_rows, all_cols = [], []
            current_idx = 0
            term_slices = {}
            
            # 1. Process terms in a strictly deterministic order.
            # We record a `slice` for each term, reserving a fixed block of 
            # 1D memory in the BCOO data array for that term's non-zeros.
            for category in ['const', 'slow', 'fast']:
                for term in self.terms_by_field[field][category]:
                    r, c = probe_term(term)
                    nnz = len(r)
                    
                    term_slices[term.uid] = {
                        'slice': slice(current_idx, current_idx + nnz),
                        # Store local rows relative to the term's output shape for fast slicing during apply()
                        'rows': r - (term.row_start if isinstance(term, _CstTerm) else 0), 
                        'cols': c
                    }
                    all_rows.append(r)
                    all_cols.append(c)
                    current_idx += nnz
            
            # 2. Append statically extracted slack indices to the end of the array.
            if field == 'P' and slack_P_rows:
                all_rows.append(np.array(slack_P_rows))
                all_cols.append(np.array(slack_P_cols))
                slack_nnz = len(slack_P_rows)
                slack_slice = slice(current_idx, current_idx + slack_nnz)
                current_idx += slack_nnz
            elif field == 'G' and slack_G_rows:
                all_rows.append(np.array(slack_G_rows))
                all_cols.append(np.array(slack_G_cols))
                slack_nnz = len(slack_G_rows)
                slack_slice = slice(current_idx, current_idx + slack_nnz)
                current_idx += slack_nnz
            else:
                slack_slice = None

            # 3. Concatenate all indices into a single (Total_NNZ, 2) XLA-compatible array.
            if all_rows:
                global_r = np.concatenate(all_rows)
                global_c = np.concatenate(all_cols)
                indices = jnp.array(np.stack([global_r, global_c], axis=1), dtype=jnp.int32)
            else:
                indices = jnp.empty((0, 2), dtype=jnp.int32)
                
            self.matrix_metadata[field] = {
                'indices': indices,
                'term_slices': term_slices,
                'total_nnz': current_idx
            }
            
            # 4. Populate the base_qp data with constant evaluations.
            data = np.zeros(current_idx, dtype=np.float64)
            for term in self.terms_by_field[field]['const']:
                out = term.fn({}) 
                meta = term_slices[term.uid]
                data[meta['slice']] = out[meta['rows'], meta['cols']]
                
            # 5. Burn slack coefficients into the constant base data permanently.
            if field == 'P' and slack_slice:
                data[slack_slice] = slack_P_vals
            elif field == 'G' and slack_slice:
                data[slack_slice] = slack_G_vals

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
            if field == 'q' and n_slack > 0:
                val[n_var : n_dec] += q_slack_vec
                
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
                out = term.fn(vars)
                
                # Extract the specific non-zeros produced by this term...
                nnz_vals = out[t_meta['rows'], t_meta['cols']]
                # ...and overwrite its dedicated 1D slice in the BCOO data array.
                new_data = new_data.at[t_meta['slice']].set(nnz_vals)
                
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

def probe_term(
    term: Union[_CostTerm, _CstTerm], 
    num_probes: int = 3, 
    key_seed: int = 42, 
    tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates a term with random inputs to find structural non-zeros.

    By evaluating the term multiple times with normally distributed random 
    inputs, we can reliably identify structurally non-zero elements while 
    avoiding "accidental zeros" that might occur with a single evaluation.
    This must be run statically at build time (outside of JIT).

    Parameters
    ----------
    term : _CostTerm or _CstTerm
        The term to probe.
    num_probes : int, optional
        Number of random evaluations to perform (default is 3).
    key_seed : int, optional
        Random seed for JAX PRNG (default is 42).
    tol : float, optional
        Tolerance below which an element is considered structurally zero.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        The global `(row_indices, col_indices)` where this term contributes 
        non-zeros to the QP matrices.
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
            
    accum_mask_np = np.array(accum_mask)
    
    # Extract 2D local indices
    if accum_mask_np.ndim == 2:
        local_rows, local_cols = np.nonzero(accum_mask_np)
    else:
        local_rows = np.nonzero(accum_mask_np)[0]
        local_cols = np.zeros_like(local_rows)
        
    # Map local structural indices to the global QP coordinates.
    # Cost terms sit in the top-left [n_var, n_var] block of P.
    # Constraint terms span [n_cst, n_var] and must be shifted by their global row.
    if isinstance(term, _CostTerm):
        return local_rows, local_cols
    else:
        return local_rows + term.row_start, local_cols