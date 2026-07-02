"""JAX-traceable solutions to the Algebraic Riccati equations."""

import jax.numpy as jnp
import scipy
import jax
import numpy as np

@jax.custom_jvp
def dare(sigma_v,sigma_w,n_y):

    # fixed covariance matrix for w
    mat1 = np.triu(np.tile(np.arange(1,n_y+1,1)[:,None],(1,n_y)))
    mat2 = np.tril(np.tile(np.arange(1,n_y+1,1),(n_y,1)),k=-1)
    covariance_base_matrix = mat1 + mat2

    # form covariance matrices
    r_w = sigma_w**2 * covariance_base_matrix
    r_v = sigma_v**2 * np.eye(n_y)

    return scipy.linalg.solve_discrete_are(np.eye(n_y),np.eye(n_y),r_w,r_v)

@dare.defjvp
def dare_jvp(primals, tangents):

    # extract inputs and tangents
    sigma_v, sigma_w, _  = primals
    sigma_v_dot, sigma_w_dot, _ = tangents

    # solve DARE
    p = dare(sigma_v,sigma_w,n_y)

    # get rhs and lhs
    _, rhs = jax.jvp(lambda s_w, s_v: dare_implicit(s_w,s_v,p,n_y),(sigma_w,sigma_v),(sigma_v_dot, sigma_w_dot))
    lhs = jax.jacobian(dare_implicit,2)(sigma_w,sigma_v,p,n_y)

    return p, -jnp.linalg.tensorsolve(lhs,rhs)


def dare_implicit(sigma_w,sigma_v,p,n_y):

    # fixed covariance matrix for w
    mat1 = jnp.triu(jnp.tile(jnp.arange(1,n_y+1,1)[:,None],(1,n_y)))
    mat2 = jnp.tril(jnp.tile(jnp.arange(1,n_y+1,1),(n_y,1)),k=-1)
    covariance_base_matrix = mat1 + mat2

    # form covariance matrices
    r_w = sigma_w**2 * covariance_base_matrix
    r_v = sigma_v**2 * jnp.eye(n_y)

    return -p @ jnp.linalg.solve(p+r_v,p) + r_w


if __name__ == '__main__':

    import time

    # covariances (scalar)
    sigma_w, sigma_v = 0.2, 0.2

    # problem dimension
    n_y = 25

    # evaluate dare
    p = dare(sigma_v,sigma_w,n_y)

    # compute jacobian
    jac_p_func = jax.jacobian(dare,(0,1))

    # evaluate once for speed
    _ = jac_p_func(sigma_v,sigma_w,n_y)

    start = time.time()
    for _ in range(20):
        jac_p = jac_p_func(sigma_v,sigma_w,n_y)
    print(f'Elapsed: {(time.time()-start)/20}')

    # compare with numerical differences
    epsilon = 1e-8
    p_v_plus = dare(sigma_v+epsilon,sigma_w,n_y)
    p_v_minus = dare(sigma_v-epsilon,sigma_w,n_y)
    p_v_diff = (p_v_plus - p_v_minus) / (2*epsilon)
    p_w_plus = dare(sigma_v,sigma_w+epsilon,n_y)
    p_w_minus = dare(sigma_v,sigma_w-epsilon,n_y)
    p_w_diff = (p_w_plus - p_w_minus) / (2*epsilon)

    # compare
    print(jnp.max(jnp.abs(p_v_diff-jac_p[1])))
    print(jnp.max(jnp.abs(p_w_diff-jac_p[0])))