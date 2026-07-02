import pytest
import jax
import jax.numpy as jnp
import numpy as np

from bpmpc.sys_id import extract_features, RLS

jax.config.update("jax_enable_x64", True)


# ======================================================================
# TEST 1: Linear Model -> Kronecker Product
# ======================================================================
def test_extract_features_linear_kronecker():
    """Verify that a purely linear parameterization f(x, u, theta) = A(theta) * x 
    results in g=0 and psi = I \\otimes x^T.
    """
    nx = 2
    n_theta = nx * nx
    
    def f_linear(x, u, theta, params):
        # theta is a flattened nx x nx matrix. Reshape to A and multiply by x.
        A = theta.reshape((nx, nx))
        return A @ x
        
    g_fn, psi_fn = extract_features(f_linear, n_theta=n_theta, verify=False)
    
    x_test = jnp.array([1.5, -0.5])
    u_test = jnp.array([0.0]) # unused
    
    g_val = g_fn(x_test, u_test)
    psi_val = psi_fn(x_test, u_test)
    
    # 1. Nominal drift g(x, u) should be strictly zero
    np.testing.assert_allclose(g_val, jnp.zeros(nx), atol=1e-7)
    
    # 2. For A @ x, the Jacobian w.r.t flattened A is exactly the 
    #    Kronecker product of the Identity matrix and x.
    #    E.g., row 0: [x_0, x_1, 0, 0], row 1: [0, 0, x_0, x_1]
    expected_psi = jnp.kron(jnp.eye(nx), x_test) 
    
    np.testing.assert_allclose(psi_val, expected_psi, atol=1e-7)


# ======================================================================
# TEST 2: Parameter-Feature Affine Model
# ======================================================================
def test_extract_features_affine():
    """Verify that a generic f(x, u) = g(x, u) + psi(x, u) @ theta is 
    recovered perfectly and passes the internal verification check.
    """
    nx = 2
    n_theta = 3
    
    # Define arbitrary complex features
    def g_true(x, u):
        return jnp.array([x[0]**2 + u[0], jnp.sin(x[1])])
        
    def psi_true(x, u):
        return jnp.array([
            [x[0], x[1], 0.0],
            [0.0,  u[0], x[0]*x[1]]
        ])
        
    # The user provides this composite function
    def f_affine(x, u, theta, params):
        return g_true(x, u) + psi_true(x, u) @ theta
        
    x_probe = jnp.array([0.5, -0.5])
    u_probe = jnp.array([1.0])
    
    # Enable verify=True to ensure the internal random check passes
    g_fn, psi_fn = extract_features(
        f_affine, n_theta=n_theta, verify=True, 
        x_probe=x_probe, u_probe=u_probe
    )
    
    g_val = g_fn(x_probe, u_probe)
    psi_val = psi_fn(x_probe, u_probe)
    
    # We should perfectly recover the underlying mathematical terms
    np.testing.assert_allclose(g_val, g_true(x_probe, u_probe), atol=1e-7)
    np.testing.assert_allclose(psi_val, psi_true(x_probe, u_probe), atol=1e-7)


# ======================================================================
# TEST 3: Fully Non-Linear Model Rejection
# ======================================================================
def test_extract_features_nonlinear_raises():
    """Verify that providing a function where theta enters non-linearly 
    (e.g., theta^2, sin(theta)) correctly triggers a ValueError during verification.
    """
    n_theta = 2
    
    def f_nonlinear(x, u, theta, params):
        # theta enters non-linearly! 
        return jnp.array([
            x[0] + jnp.sin(theta[0]),   # Non-linear 1
            u[0] + theta[1]**2          # Non-linear 2
        ])
        
    x_probe = jnp.array([1.0, 2.0])
    u_probe = jnp.array([0.5])
    
    # The function should trap the invalid parameterization and raise an exception
    with pytest.raises(ValueError, match="Dynamics function is not affine in theta"):
        extract_features(
            f_nonlinear, n_theta=n_theta, verify=True, 
            x_probe=x_probe, u_probe=u_probe
        )

# ======================================================================
# TEST 4: RLS Identification of a Linear System
# ======================================================================
def test_rls_linear_identification():
    """Verify that RLS can perfectly identify the elements of a dense A matrix 
    given persistently exciting states.
    """
    nx, nu = 2, 1
    n_theta = nx * nx
    
    A_true = jnp.array([[0.8, 0.2], [-0.1, 0.9]])
    theta_true = A_true.flatten()
    
    def f_linear(x, u, theta, params):
        A = theta.reshape((nx, nx))
        return A @ x

    g_fn, psi_fn = extract_features(f_linear, n_theta)
    rls = RLS(n_theta=n_theta, g_fn=g_fn, psi_fn=psi_fn, lam=1.0)
    
    # 1. Initialize RLS state with low confidence
    state = rls.init(theta0=jnp.zeros(n_theta), P0_inv=jnp.eye(n_theta) * 1e-4)
    
    # 2. Generate a random trajectory for persistence of excitation
    key = jax.random.PRNGKey(0)
    N_samples = 20
    X = jax.random.normal(key, (N_samples, nx))
    U = jnp.zeros((N_samples, nu))
    
    # 3. Sequentially update the RLS estimator
    for i in range(N_samples):
        x_curr = X[i]
        u_curr = U[i]
        x_next = A_true @ x_curr
        
        state = rls.step(state, x_curr, u_curr, x_next)
        
    # 4. Assert convergence to true parameters
    np.testing.assert_allclose(state.theta, theta_true, atol=1e-5)


# ======================================================================
# TEST 5: RLS Identification of a Non-Linear Affine System
# ======================================================================
def test_rls_nonlinear_affine_identification():
    """Verify that RLS can perfectly identify parameters in a complex, non-linear
    parameter-affine system f(x, u) = g(x, u) + psi(x, u)*theta.
    """
    nx, nu = 2, 1
    n_theta = 2
    theta_true = jnp.array([0.5, -1.5])
    
    # Non-linear dynamics
    def f_nonlinear(x, u, theta, params):
        g = jnp.array([x[0]**2 + jnp.cos(u[0]), x[1]])
        psi = jnp.array([
            [x[0] * x[1], 0.0],
            [0.0,         jnp.sin(x[0])]
        ])
        return g + psi @ theta

    g_fn, psi_fn = extract_features(f_nonlinear, n_theta)
    rls = RLS(n_theta=n_theta, g_fn=g_fn, psi_fn=psi_fn, lam=1.0)
    
    state = rls.init(theta0=jnp.zeros(n_theta), P0_inv=jnp.eye(n_theta) * 1e-4)
    
    key = jax.random.PRNGKey(1)
    N_samples = 100
    X = jax.random.normal(key, (N_samples, nx))
    U = jax.random.normal(key, (N_samples, nu))
    
    for i in range(N_samples):
        x_curr = X[i]
        u_curr = U[i]
        # Generate true observation
        x_next = f_nonlinear(x_curr, u_curr, theta_true, {})
        
        state = rls.step(state, x_curr, u_curr, x_next)
        
    np.testing.assert_allclose(state.theta, theta_true, atol=1e-5)


# ======================================================================
# TEST 6: Equivalence of Batched (Block) vs Sequential RLS
# ======================================================================
def test_rls_batched_vs_sequential():
    """Verify that passing an entire trajectory of inputs as a batched matrix 
    (Block RLS) produces the exact same Information matrix and parameter estimate 
    as iterating through the trajectory one step at a time.
    """
    nx, nu = 2, 1
    n_theta = 2
    
    # Simple affine model for testing
    def f_test(x, u, theta, params):
        return x + jnp.array([[x[0], u[0]], [u[0], x[1]]]) @ theta
        
    g_fn, psi_fn = extract_features(f_test, n_theta)
    
    # Note: For strict equivalence without discounting past data within the block, 
    # lambda must be exactly 1.0 (infinite memory).
    rls = RLS(n_theta=n_theta, g_fn=g_fn, psi_fn=psi_fn, lam=1.0)
    
    # 1. Base initialization
    theta0 = jnp.zeros(n_theta)
    P0_inv = jnp.eye(n_theta) * 0.1
    state0 = rls.init(theta0, P0_inv)
    
    # 2. Generate a random trajectory chunk
    key = jax.random.PRNGKey(2)
    N_batch = 10
    X = jax.random.normal(key, (N_batch, nx))
    U = jax.random.normal(key, (N_batch, nu))
    
    # Generate target measurements using arbitrary true parameters
    theta_true = jnp.array([1.0, -1.0])
    X_next = jax.vmap(lambda x, u: f_test(x, u, theta_true, {}))(X, U)
    
    # 3. Perform Batched (Block) Update
    state_batched = rls.step(state0, X, U, X_next)
    
    # 4. Perform Sequential (For-loop) Update
    state_sequential = state0
    for i in range(N_batch):
        state_sequential = rls.step(state_sequential, X[i], U[i], X_next[i])
        
    # 5. The final states must be mathematically identical
    np.testing.assert_allclose(state_batched.P_inv, state_sequential.P_inv, atol=1e-7)
    np.testing.assert_allclose(state_batched.theta, state_sequential.theta, atol=1e-7)