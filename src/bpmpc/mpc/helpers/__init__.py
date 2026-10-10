"""Convenience factories and builders for common MPC building blocks.

**Factories** return ready-to-use ``Constraint`` / ``Cost`` objects.
Arguments accept ``Array`` (constant) or ``Variable`` (parametric).

**Builders** are pure functions that take concrete arrays and return
matrices.  Useful for custom parametric logic inside a lambda.  Matrix
builders return a ``BCOO`` with static indices by default, which a
``Cost`` / ``Constraint`` uses as its sparsity pattern directly; pass
``sparse=False`` to get a dense array.

Usage::

    from bpmpc_jax.mpc.helpers import (
        # Factories (return Constraint / Cost)
        lti_dynamics, ltv_dynamics,
        box_bounds,
        state_tracking_cost, output_tracking_cost,

        # Builders (return arrays)
        build_lti_lhs, build_lti_rhs,
        build_ltv_lhs, build_ltv_rhs,
        build_box_lhs, build_box_rhs,
        build_state_tracking, build_output_tracking,
    )
"""

from ._src.dynamics import (
    lti_dynamics, ltv_dynamics,
    build_lti_lhs, build_lti_rhs,
    build_ltv_lhs, build_ltv_rhs,
    nonlinear_dynamics
)
from ._src.bounds import (
    box_bounds,
    build_box_lhs, build_box_rhs,
)
from ._src.costs import (
    state_tracking_cost, output_tracking_cost,
    build_state_tracking, build_output_tracking,
)

__all__ = [
    # Factories
    "lti_dynamics", "ltv_dynamics", "nonlinear_dynamics",
    "box_bounds",
    "state_tracking_cost", "output_tracking_cost",
    # Builders
    "build_lti_lhs", "build_lti_rhs",
    "build_ltv_lhs", "build_ltv_rhs",
    "build_box_lhs", "build_box_rhs",
    "build_state_tracking", "build_output_tracking",
]