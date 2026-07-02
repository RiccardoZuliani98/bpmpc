"""Discrete-time dynamical system with a straight-through gradient estimator.

This module provides a mechanism to decouple the forward simulation of a 
dynamical system from its automatic differentiation (AD) paths. 
The forward pass evaluates the ``true_fun`` (representing the real, 
potentially non-differentiable or noisy system). The backward or forward-mode 
AD pass evaluates the ``nominal_fun`` (representing a differentiable surrogate model).
The two functions may depend on completely disjoint parameter dictionaries.
"""

from __future__ import annotations
from typing import Callable, Mapping, Sequence
from functools import partial

import jax
import jax.numpy as jnp

from bpmpc.variable import Variable


# ----------------------------------------------------------------------
# Core primitive: forward = true_fun, AD = nominal_fun
# ----------------------------------------------------------------------
@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _straight_through_step(
    true_fun: Callable,
    nominal_fun: Callable,
    state: jax.Array,
    action: jax.Array,
    true_params: Mapping[str, jax.Array],
    nominal_params: Mapping[str, jax.Array],
) -> jax.Array:
    """Evaluate the primal pass using strictly the true system dynamics.
    
    The ``nominal_fun`` is ignored during the forward execution.
    """
    return true_fun(state, action, true_params)


@_straight_through_step.defjvp
def _st_jvp(true_fun, nominal_fun, primals, tangents):
    """Compute the Jacobian-vector product (JVP) for the straight-through step.
    
    Routes the primal calculation through the ``true_fun``, but routes all 
    tangent (gradient) calculations through the ``nominal_fun``.
    """
    state, action, true_params, nominal_params = primals
    dstate, daction, _dtrue_params, dnominal_params = tangents

    # Primal output from the true system.
    primal_out = true_fun(state, action, true_params)

    # Tangent output from the nominal model; tangent on true_params is ignored.
    _, tangent_out = jax.jvp(
        nominal_fun,
        (state, action, nominal_params),
        (dstate, daction, dnominal_params),
    )
    return primal_out, tangent_out


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
class Dynamics:
    """Discrete-time system that is truthful in value and model-based in gradient.

    This class encapsulates a dynamical system for model predictive control or 
    reinforcement learning where the true environment differs from the internal 
    planning model. Each dynamics function has the signature 
    ``f(state, action, params) -> next_state``, where ``params`` is a dictionary 
    mapping variable names to JAX arrays.

    Parameters
    ----------
    true_fun : Callable
        The true (forward-pass) dynamics. Represents the actual environment 
        and may be noisy, complex, or non-differentiable.
    nx : int
        The dimension (size) of the system state vector.
    nu : int
        The dimension (size) of the system control input (action) vector.
    nominal_fun : Callable, optional
        The nominal (AD-pass) model used for computing gradients. Must be 
        differentiable. Defaults to ``true_fun``.
    true_params_spec : Sequence[Variable], optional
        Sequence of ``Variable`` objects describing the parameter dictionary 
        expected by ``true_fun``. Used for trace-time validation. Defaults to ().
    nominal_params_spec : Sequence[Variable], optional
        Sequence of ``Variable`` objects describing the parameter dictionary 
        expected by ``nominal_fun``. Defaults to ``true_params_spec`` when 
        ``nominal_fun`` is also defaulted; otherwise defaults to ().

    Notes
    -----
    * Works seamlessly with ``jax.jit``, ``jax.grad``, ``jax.vjp``, ``jax.jvp``,
      ``jax.jacfwd``, and ``jax.jacrev``.
    * In forward-only mode (e.g., no AD trace active), ``nominal_fun`` is 
      never evaluated, saving compute.
    * No gradient flows to ``true_params`` — it represents external, unchangeable reality.
    * Both ``true_fun`` and ``nominal_fun`` must be hashable. Standard functions 
      are hashable, but lambdas capturing JAX arrays are not. Pass dynamic arrays 
      via the ``params`` dictionaries instead.
    """

    def __init__(
        self,
        true_fun: Callable,
        nx: int,
        nu: int,
        nominal_fun: Callable | None = None,
        true_params_spec: Sequence[Variable] = (),
        nominal_params_spec: Sequence[Variable] | None = None,
    ) -> None:
        self.true_fun = true_fun
        self.nx = nx
        self.nu = nu
        self.nominal_fun = nominal_fun if nominal_fun is not None else true_fun

        if nominal_params_spec is None:
            nominal_params_spec = true_params_spec if nominal_fun is None else ()

        self.true_params_spec: dict[str, Variable] = {
            v.name: v for v in true_params_spec
        }
        self.nominal_params_spec: dict[str, Variable] = {
            v.name: v for v in nominal_params_spec
        }

    # ------------------------------------------------------------------
    # Validation (trace-time only; zero runtime cost after jit)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(
        params: Mapping[str, jax.Array],
        spec: Mapping[str, Variable],
        which: str,
    ) -> None:
        """Validates parameter dictionaries against their Variable specifications."""
        missing = set(spec) - set(params)
        if missing:
            raise KeyError(f"{which}_params missing keys: {sorted(missing)}")
        extra = set(params) - set(spec)
        if extra:
            raise KeyError(f"{which}_params has unexpected keys: {sorted(extra)}")
        for name, var in spec.items():
            shape = jnp.shape(params[name])
            if shape != var.shape:
                raise ValueError(
                    f"{which}_params[{name!r}] has shape {shape}, "
                    f"expected {var.shape}"
                )

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def step(
        self,
        state: jax.Array,
        action: jax.Array,
        true_params: Mapping[str, jax.Array] | None = None,
        nominal_params: Mapping[str, jax.Array] | None = None,
    ) -> jax.Array:
        """Advance the system forward by one discrete time step.

        The resulting state value equals ``true_fun(state, action, true_params)``. 
        However, any AD passes (forward or reverse mode) will route gradients 
        through ``nominal_fun(state, action, nominal_params)``.

        Parameters
        ----------
        state : jax.Array
            The current state of the system.
        action : jax.Array
            The control input applied to the system.
        true_params : Mapping[str, jax.Array], optional
            Dictionary of parameters required by the true environment function.
        nominal_params : Mapping[str, jax.Array], optional
            Dictionary of parameters required by the nominal model function.

        Returns
        -------
        jax.Array
            The next state of the system.
        """
        # Validate state and action trailing dimensions (vmap-compatible check)
        if state.shape[-1] != self.nx:
            raise ValueError(
                f"Invalid state dimension: expected trailing axis of size {self.nx}, "
                f"but got shape {state.shape}."
            )
        if action.shape[-1] != self.nu:
            raise ValueError(
                f"Invalid action dimension: expected trailing axis of size {self.nu}, "
                f"but got shape {action.shape}."
            )

        true_params = dict(true_params) if true_params else {}
        nominal_params = dict(nominal_params) if nominal_params else {}

        if self.true_params_spec:
            self._validate(true_params, self.true_params_spec, "true")
        if self.nominal_params_spec:
            self._validate(nominal_params, self.nominal_params_spec, "nominal")

        return _straight_through_step(
            self.true_fun,
            self.nominal_fun,
            state,
            action,
            true_params,
            nominal_params,
        )

    def rollout(
        self,
        state0: jax.Array,
        actions: jax.Array,  # shape (T, *action_shape)
        true_params: Mapping[str, jax.Array] | None = None,
        nominal_params: Mapping[str, jax.Array] | None = None,
    ) -> jax.Array:
        """Simulate the system over a sequence of actions via ``jax.lax.scan``.

        Parameters
        ----------
        state0 : jax.Array
            The initial state of the system.
        actions : jax.Array
            A sequence of control inputs to apply, with shape ``(T, *action_shape)`` 
            where ``T`` is the time horizon.
        true_params : Mapping[str, jax.Array], optional
            Dictionary of parameters required by the true environment function.
        nominal_params : Mapping[str, jax.Array], optional
            Dictionary of parameters required by the nominal model function.

        Returns
        -------
        jax.Array
            The complete state trajectory including the initial state, 
            yielding a shape of ``(T + 1, *state_shape)``.
        """
        def body(state, u):
            next_state = self.step(state, u, true_params, nominal_params)
            return next_state, next_state

        _, trajectory = jax.lax.scan(body, state0, actions)
        return jnp.concatenate([state0[None], trajectory], axis=0)