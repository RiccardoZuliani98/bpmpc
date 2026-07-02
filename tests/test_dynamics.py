"""Tests for Dynamics with straight-through gradient estimator."""

import jax
import jax.numpy as jnp
import pytest

from bpmpc.dynamics import Dynamics
from bpmpc.variable import Variable

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------
def make_funs():
    """A true system that doubles the action; a nominal model with a `gain`."""
    def true_fun(state, action, params):
        return state + 2.0 * action

    def nominal_fun(state, action, params):
        return state + params["gain"] * action

    return true_fun, nominal_fun


def make_dynamics():
    true_fun, nominal_fun = make_funs()
    return Dynamics(
        true_fun=true_fun,
        nominal_fun=nominal_fun,
        nx=NX,nu=NU,
        true_params_spec=(),
        nominal_params_spec=(Variable("gain", (1,)),),
    )


STATE = jnp.array([0.0, 0.0])
ACTION = jnp.array([1.0, 1.0])
NX, NU = 2, 2
NOMINAL_PARAMS = {"gain": jnp.array([3.0])}  # deliberately != 2.0


# ----------------------------------------------------------------------
# Value correctness
# ----------------------------------------------------------------------
def test_forward_value_matches_true_fun():
    dyn = make_dynamics()
    out = dyn.step(STATE, ACTION, {}, NOMINAL_PARAMS)
    # true_fun: state + 2 * action = [2, 2]
    assert jnp.allclose(out, jnp.array([2.0, 2.0]))


def test_forward_only_does_not_call_nominal():
    # Define a normal true function
    def true_fun(state, action, params):
        return state + 2.0 * action
    
    # Define a poisonous nominal function
    def poisonous_nominal_fun(state, action, params):
        raise RuntimeError("JAX attempted to evaluate nominal_fun during a forward pass!")

    dyn = Dynamics(
        true_fun=true_fun,
        nominal_fun=poisonous_nominal_fun,
        nx=NX,nu=NU,
        true_params_spec=(),
        nominal_params_spec=(Variable("gain", (1,)),),
    )
    
    # If the straight-through estimator works correctly, this will run smoothly.
    # If it accidentally routes through nominal_fun, the test will crash.
    dyn.step(STATE, ACTION, {}, NOMINAL_PARAMS)


# ----------------------------------------------------------------------
# Reverse-mode AD
# ----------------------------------------------------------------------
def test_grad_routes_through_nominal():
    dyn = make_dynamics()

    def loss(nominal_params):
        return dyn.step(STATE, ACTION, {}, nominal_params).sum()

    g = jax.grad(loss)(NOMINAL_PARAMS)
    # d/dgain [ sum(state + gain * action) ] = sum(action) = 2.0
    assert jnp.allclose(g["gain"], jnp.array([2.0]))


def test_grad_wrt_state_uses_nominal():
    dyn = make_dynamics()

    def f(state):
        # Nominal Jacobian wrt state is identity -> grad of sum is ones.
        return dyn.step(state, ACTION, {}, NOMINAL_PARAMS).sum()

    g = jax.grad(f)(STATE)
    assert jnp.allclose(g, jnp.ones_like(STATE))


# ----------------------------------------------------------------------
# Forward-mode AD
# ----------------------------------------------------------------------
def test_jvp_uses_nominal_for_tangent():
    dyn = make_dynamics()

    def f(nominal_params):
        return dyn.step(STATE, ACTION, {}, nominal_params)

    tangent_in = {"gain": jnp.array([1.0])}
    primal, tangent = jax.jvp(f, (NOMINAL_PARAMS,), (tangent_in,))

    # Primal from true_fun: [2, 2].
    assert jnp.allclose(primal, jnp.array([2.0, 2.0]))
    # Tangent from nominal_fun: d/dgain (gain * action) * 1 = action = [1, 1].
    assert jnp.allclose(tangent, jnp.array([1.0, 1.0]))


# ----------------------------------------------------------------------
# jit compatibility
# ----------------------------------------------------------------------
def test_jit_forward():
    dyn = make_dynamics()
    jitted = jax.jit(lambda s, a, np_: dyn.step(s, a, {}, np_))
    out = jitted(STATE, ACTION, NOMINAL_PARAMS)
    assert jnp.allclose(out, jnp.array([2.0, 2.0]))


def test_jit_grad():
    dyn = make_dynamics()

    @jax.jit
    def grad_loss(nominal_params):
        return jax.grad(lambda p: dyn.step(STATE, ACTION, {}, p).sum())(nominal_params)

    g = grad_loss(NOMINAL_PARAMS)
    assert jnp.allclose(g["gain"], jnp.array([2.0]))


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------
def test_rollout_shape_and_values():
    dyn = make_dynamics()
    actions = jnp.ones((4, 2))
    traj = dyn.rollout(STATE, actions, {}, NOMINAL_PARAMS)

    # Shape: T + 1 = 5.
    assert traj.shape == (5, 2)
    # Each step adds 2 * action = [2, 2] via true_fun.
    expected = jnp.stack([jnp.array([2.0 * i, 2.0 * i]) for i in range(5)])
    assert jnp.allclose(traj, expected)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_validation_missing_key():
    dyn = make_dynamics()
    with pytest.raises(KeyError, match="missing keys"):
        dyn.step(STATE, ACTION, {}, {})  # gain missing


def test_validation_wrong_shape():
    dyn = make_dynamics()
    with pytest.raises(ValueError, match="shape"):
        dyn.step(STATE, ACTION, {}, {"gain": jnp.array([1.0, 2.0])})  # (2,) != (1,)


def test_validation_extra_key():
    dyn = make_dynamics()
    with pytest.raises(KeyError, match="unexpected keys"):
        dyn.step(STATE, ACTION, {}, {"gain": jnp.array([1.0]), "extra": jnp.array([0.0])})


# ----------------------------------------------------------------------
# Disjoint parameter sets
# ----------------------------------------------------------------------
def test_disjoint_params_namespaces():
    """true and nominal can use different parameter names without collision."""

    def true_fun(state, action, params):
        return state + params["mass_true"] * action

    def nominal_fun(state, action, params):
        return state + params["mass_model"] * action

    dyn = Dynamics(
        true_fun=true_fun,
        nominal_fun=nominal_fun,
        nx=NX,nu=NU,
        true_params_spec=(Variable("mass_true", (1,)),),
        nominal_params_spec=(Variable("mass_model", (1,)),),
    )

    out = dyn.step(
        STATE, ACTION,
        {"mass_true": jnp.array([5.0])},
        {"mass_model": jnp.array([1.0])},
    )
    # Forward uses true mass = 5.
    assert jnp.allclose(out, jnp.array([5.0, 5.0]))