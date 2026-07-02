"""
Pick2Learn: greedy curriculum builder.

Library-agnostic. Pick2Learn knows about three things:

    * ``thetas_pool`` : pytree of arrays, each with leading dim S
                        (e.g. ``{"x0": (S, NX) array, "mass": (S,) array}``);
                        a single ``(S, n_theta)`` array also works.
    * ``loss_fn``     : (params, thetas) -> (n,) vmapped per-scenario loss
    * ``optimizer``   : (params, thetas) -> (new_params, log)
                        runs until its own embedded stopping criterion

In both ``loss_fn`` and ``optimizer``, ``thetas`` is a pytree with the same
structure as ``thetas_pool`` but with leading dim n (the size of the subset
being evaluated / trained on).

Per round:
    1. From the current snapshot, pick the worst-loss scenario in remaining.
    2. Move it to working.
    3. Run the optimizer on the working set.
    4. Re-evaluate the full pool with the new params.
    5. Stop when remaining is empty or no remaining scenario exceeds the
       worst working loss.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


# Type aliases (purely documentary)
Params    = Any
ThetaPool = Any  # pytree of arrays, each with leading dim S
LossFn    = Callable[[Params, ThetaPool], jnp.ndarray]
Optimizer = Callable[[Params, ThetaPool], "tuple[Params, Any]"]


# ════════════════════════════════════════════════════════════════════════
#  Pool helpers
# ════════════════════════════════════════════════════════════════════════

def stack_thetas(thetas_list: Sequence[Any]) -> Any:
    """Convert a list of pytrees into a single pytree of stacked arrays.

    Each element of ``thetas_list`` must have the same structure. Leaves
    are stacked along a new leading axis of size ``len(thetas_list)``.

    Convenient for authoring scenarios as a list of dicts, then handing
    the result to :class:`Pick2Learn`.

    Examples
    --------
    >>> thetas = stack_thetas([
    ...     {"x0": jnp.zeros(4), "mass": jnp.array(1.0)},
    ...     {"x0": jnp.ones(4),  "mass": jnp.array(2.0)},
    ... ])
    >>> thetas["x0"].shape, thetas["mass"].shape
    ((2, 4), (2,))
    """
    if len(thetas_list) == 0:
        raise ValueError("thetas_list is empty.")
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *thetas_list)


def _pool_size(pool: ThetaPool) -> int:
    """Return S, validating that every leaf has the same leading dim."""
    leaves = jax.tree_util.tree_leaves(pool)
    if not leaves:
        raise ValueError("thetas_pool has no leaves.")
    sizes = {int(leaf.shape[0]) for leaf in leaves}
    if len(sizes) != 1:
        raise ValueError(
            f"thetas_pool leaves have inconsistent leading dimensions: {sizes}"
        )
    return sizes.pop()


def _index_pool(pool: ThetaPool, idx: jnp.ndarray) -> ThetaPool:
    """Index every leaf of ``pool`` along its leading axis with ``idx``."""
    return jax.tree_util.tree_map(lambda leaf: leaf[idx], pool)


# ════════════════════════════════════════════════════════════════════════
#  P2LState — snapshot at the boundary of a round
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class P2LState:
    """Snapshot at the boundary of a round.

    Attributes
    ----------
    working_idx : list[int]
        Pool indices in the working set, in insertion order.
    remaining_idx : list[int]
        Pool indices not yet picked.
    params : pytree
        Parameters that produced this snapshot's losses.
    losses : np.ndarray, shape (S,)
        Per-pool-element losses evaluated at ``params``.
    training : Any, optional
        Opaque training log returned by the optimizer for the round
        that produced this state. ``None`` for a non-seeded initial state.
    """
    working_idx:   List[int]
    remaining_idx: List[int]
    params:        Params
    losses:        np.ndarray
    training:      Optional[Any] = None

    @property
    def worst_working_loss(self) -> float:
        if not self.working_idx:
            return -np.inf
        return float(np.max(self.losses[self.working_idx]))

    @property
    def worst_remaining_loss(self) -> float:
        if not self.remaining_idx:
            return -np.inf
        return float(np.max(self.losses[self.remaining_idx]))


# ════════════════════════════════════════════════════════════════════════
#  P2LResult — list of states + reproducibility metadata
# ════════════════════════════════════════════════════════════════════════

@dataclass
class P2LResult:
    """Output of a Pick2Learn run.

    Attributes
    ----------
    states : list of P2LState
        ``states[0]`` is the initial snapshot (after any seed optimisation).
        ``states[k]`` for k >= 1 is the snapshot after round ``k``.
    thetas_pool : np.ndarray, (S, n_theta)
        The candidate pool used.
    config : dict
        Free-form hyperparameter record.
    terminated_early : bool
        True if the loop stopped because no remaining loss exceeded the
        worst working loss; False if the pool was fully exhausted.
    timestamp : str
    notes : str
    """
    states:           List[P2LState]
    thetas_pool:      ThetaPool
    config:           dict
    terminated_early: bool
    timestamp:        str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    )
    notes:            str = ""

    # ── Convenience ──────────────────────────────────────────────────

    @property
    def initial_state(self) -> P2LState:
        return self.states[0]

    @property
    def final_state(self) -> P2LState:
        return self.states[-1]

    @property
    def n_rounds(self) -> int:
        """Number of pick rounds (excluding the initial snapshot)."""
        return len(self.states) - 1

    def picked_indices(self) -> List[int]:
        """Pool indices in the order they were picked across rounds.

        Excludes any seed indices (they're already in ``states[0]``).
        """
        out: List[int] = []
        for prev, curr in zip(self.states[:-1], self.states[1:]):
            new = set(curr.working_idx) - set(prev.working_idx)
            if len(new) != 1:
                raise RuntimeError(
                    f"Expected exactly one new index between consecutive "
                    f"states; got {new}."
                )
            out.append(next(iter(new)))
        return out

    # ── Save ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save the result to ``path``.

        Layout::

            path/
                meta.json     # config, timestamp, indices, scalar summaries
                arrays.npz    # thetas_pool, per-state losses, params leaves
                training.pkl  # list of opaque training logs (one per state)

        The training logs are pickled because their type is opaque to
        Pick2Learn. The structured everything-else lives in JSON/NPZ.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # ── Arrays ────────────────────────────────────────────────────
        arrays: dict[str, np.ndarray] = {
            "losses": np.stack([s.losses for s in self.states]),
        }

        # Pool: flatten the pytree like we do for params.
        pool_leaves, pool_treedef = jax.tree_util.tree_flatten(self.thetas_pool)
        for j, leaf in enumerate(pool_leaves):
            arrays[f"thetas_pool_leaf_{j:03d}"] = np.asarray(leaf)
        pool_treedef_repr = str(pool_treedef)

        # Params: flatten each state's pytree into leaves and store as
        # individual arrays. Treedef is stored as a string for inspection
        # only — full reconstruction requires the user's original pytree
        # structure.
        params_treedef_repr: Optional[str] = None
        leaf_counts: List[int] = []
        for k, s in enumerate(self.states):
            leaves, treedef = jax.tree_util.tree_flatten(s.params)
            if params_treedef_repr is None:
                params_treedef_repr = str(treedef)
            for j, leaf in enumerate(leaves):
                arrays[f"params_state_{k:03d}_leaf_{j:03d}"] = np.asarray(leaf)
            leaf_counts.append(len(leaves))

        np.savez(path / "arrays.npz", **arrays)

        # ── Training logs ─────────────────────────────────────────────
        with open(path / "training.pkl", "wb") as f:
            pickle.dump([s.training for s in self.states], f)

        # ── Metadata ──────────────────────────────────────────────────
        meta = {
            "config":           self.config,
            "timestamp":        self.timestamp,
            "notes":            self.notes,
            "terminated_early": self.terminated_early,
            "n_states":         len(self.states),
            "n_rounds":             self.n_rounds,
            "pool_treedef_repr":    pool_treedef_repr,
            "n_pool_leaves":        len(pool_leaves),
            "params_treedef_repr":  params_treedef_repr,
            "leaves_per_state":     leaf_counts,
            "states": [
                {
                    "working_idx":          s.working_idx,
                    "remaining_idx":        s.remaining_idx,
                    "worst_working_loss":   s.worst_working_loss,
                    "worst_remaining_loss": s.worst_remaining_loss,
                }
                for s in self.states
            ],
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return path

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> str:
        n_pool = _pool_size(self.thetas_pool)
        s0, sN = self.initial_state, self.final_state

        lines = [
            f"Pick2Learn  {self.timestamp}"
            + (f"  |  {self.notes}" if self.notes else ""),
            f"Config: {self.config}",
            f"Pool: {n_pool}  |  Final working: {len(sN.working_idx)}  "
            f"|  Rounds: {self.n_rounds}  "
            f"|  Early stop: {self.terminated_early}",
            f"Worst working loss: {s0.worst_working_loss:.3e} → "
            f"{sN.worst_working_loss:.3e}",
            "",
        ]
        for k, s in enumerate(self.states):
            tag = "init" if k == 0 else f"r{k:2d}"
            lines.append(
                f"  [{tag}]  working={len(s.working_idx):3d}  "
                f"remaining={len(s.remaining_idx):3d}  "
                f"worst_w={s.worst_working_loss:.3e}  "
                f"worst_r={s.worst_remaining_loss:.3e}"
            )
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
#  Pick2Learn
# ════════════════════════════════════════════════════════════════════════

class Pick2Learn:
    """Greedy curriculum builder.

    Parameters
    ----------
    loss_fn : (params, thetas) -> jnp.ndarray of shape (n,)
        Vmapped per-scenario loss. ``thetas`` is a pytree with the same
        structure as ``thetas_pool`` but with leading dim n. Returns a
        vector of n scalar losses.
    optimizer : (params, thetas) -> (new_params, training_log)
        Callable that runs an optimisation on the given working-set
        ``thetas`` until its own (embedded) stopping criterion is met.
        Returns the resulting params and an opaque training log (stored
        verbatim in the produced state).
    thetas_pool : pytree of arrays
        Each leaf has leading dim S (the pool size). A plain
        ``(S, n_theta)`` array works; the typical shape is
        ``{"x0": (S, NX) array, "mass": (S,) array, ...}``. Use
        :func:`stack_thetas` to convert from a list of dicts.
    verbose : bool
        Whether to print per-round progress in :meth:`run`.

    Examples
    --------
    >>> pool = stack_thetas([dict_a, dict_b, dict_c])
    >>> p2l = Pick2Learn(loss_fn, optimizer, pool)
    >>> result = p2l.run(params_init, working_idx=[0])
    >>> result.save("results/run_001")
    """

    def __init__(
        self,
        loss_fn: LossFn,
        optimizer: Optimizer,
        thetas_pool: ThetaPool,
        verbose: bool = True,
    ) -> None:
        self._loss_fn     = loss_fn
        self._optimizer   = optimizer
        self._thetas_pool = thetas_pool
        self._pool_size   = _pool_size(thetas_pool)  # validates consistency
        self._verbose     = verbose

    @property
    def pool_size(self) -> int:
        return self._pool_size

    # ── Build the initial snapshot ───────────────────────────────────

    def initial_state(
        self,
        params: Params,
        working_idx: Optional[List[int]] = None,
    ) -> P2LState:
        """Build state 0.

        If ``working_idx`` is non-empty the optimiser is run once on those
        scenarios; otherwise the snapshot uses ``params`` as given. Note
        that this computes the first set of losses no matter what.
        """
        S = self.pool_size
        working = list(working_idx or [])
        if any(i < 0 or i >= S for i in working):
            raise ValueError(
                f"working_idx contains invalid indices for pool size {S}."
            )
        if len(set(working)) != len(working):
            raise ValueError("working_idx contains duplicate indices.")

        remaining = [i for i in range(S) if i not in set(working)]

        if working:
            thetas_w = _index_pool(self._thetas_pool, jnp.array(working))
            params, training = self._optimizer(params, thetas_w)
        else:
            training = None

        losses = np.asarray(self._loss_fn(params, self._thetas_pool))

        return P2LState(
            working_idx   = working,
            remaining_idx = remaining,
            params        = params,
            losses        = losses,
            training      = training,
        )

    # ── One round ────────────────────────────────────────────────────

    def step(self, state: P2LState) -> Optional[P2LState]:
        """Execute one pick round, or return ``None`` if done.

        Termination criteria (either is sufficient):
            * ``state.remaining_idx`` is empty (pool exhausted), or
            * no remaining scenario exceeds the worst working loss.
        """
        if not state.remaining_idx:
            return None

        worst_r = state.worst_remaining_loss
        worst_w = state.worst_working_loss
        if state.working_idx and worst_r <= worst_w:
            return None

        # Pick the worst-loss remaining
        rem        = np.asarray(state.remaining_idx)
        rem_losses = state.losses[rem]
        picked     = int(rem[int(np.argmax(rem_losses))])

        new_working   = state.working_idx + [picked]
        new_remaining = [i for i in state.remaining_idx if i != picked]

        # Optimise on the new working set, then re-evaluate the full pool
        thetas_w  = _index_pool(self._thetas_pool, jnp.array(new_working))
        new_params, training = self._optimizer(state.params, thetas_w)
        new_losses = np.asarray(self._loss_fn(new_params, self._thetas_pool))

        return P2LState(
            working_idx   = new_working,
            remaining_idx = new_remaining,
            params        = new_params,
            losses        = new_losses,
            training      = training,
        )

    # ── Iterator over states ─────────────────────────────────────────

    def iter_states(
        self,
        params: Params,
        working_idx: Optional[List[int]] = None,
    ) -> Iterator[P2LState]:
        """Yield successive states until termination.

        Useful when the caller wants to drive the loop manually, attach
        external stopping conditions, or stream results to a database.
        """
        state = self.initial_state(params, working_idx)
        yield state
        while True:
            nxt = self.step(state)
            if nxt is None:
                return
            state = nxt
            yield state

    # ── Full driver ──────────────────────────────────────────────────

    def run(
        self,
        params: Params,
        working_idx: Optional[List[int]] = None,
        config: Optional[dict] = None,
        notes: str = "",
    ) -> P2LResult:
        """Execute the full pick-to-learn loop until termination.

        Parameters
        ----------
        params : pytree
            Initial parameters.
        working_idx : list[int], optional
            Pool indices to seed the working set with. If supplied, the
            optimiser runs once on these scenarios before round 1.
        config : dict, optional
            Hyperparameters / metadata to attach to the result.
        notes : str
            Free-form annotation.
        """
        if self._verbose:
            print("=" * 65)
            print(f"Pick2Learn  |  pool = {self.pool_size}")
            if working_idx:
                print(f"  Seed working set: {working_idx}")
            print("=" * 65)

        states: List[P2LState] = []
        for k, state in enumerate(self.iter_states(params, working_idx)):
            states.append(state)
            if self._verbose:
                tag = "init" if k == 0 else f"r{k:2d}"
                self._print_state(tag, state)

        terminated_early = len(states[-1].remaining_idx) > 0
        if self._verbose:
            tag = "early stop" if terminated_early else "pool exhausted"
            print("=" * 65)
            print(f"Pick2Learn done  |  {len(states) - 1} rounds  |  "
                  f"working = {len(states[-1].working_idx)}  |  {tag}")
            print("=" * 65)

        return P2LResult(
            states           = states,
            thetas_pool      = jax.tree_util.tree_map(np.asarray, self._thetas_pool),
            config           = config or {},
            terminated_early = terminated_early,
            notes            = notes,
        )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _print_state(tag: str, s: P2LState) -> None:
        print(f"  [{tag}]  working={len(s.working_idx):3d}  "
              f"remaining={len(s.remaining_idx):3d}  "
              f"worst_w={s.worst_working_loss:.3e}  "
              f"worst_r={s.worst_remaining_loss:.3e}")