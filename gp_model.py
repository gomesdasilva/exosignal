import matplotlib.pylab as plt
import numpy as np

import functools
import jax
import jax.numpy as jnp

import logging
logger = logging.getLogger()



# Gram Matrix:
def gram(kernel: callable, params: dict, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda x1: jax.vmap(lambda y1: kernel(params, x1, y1))(y))(x)


def _cov_func(kernel: callable) -> callable:
    """Make convariance matrix function."""
    return functools.partial(gram, kernel)

# not used
def cov_matrix(kernel: callable, theta: dict, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    cov_f = _cov_func(kernel)
    return cov_f(theta, x, y)


# compute model:
def gp_prior(theta: dict, mean_f: callable, cov_f: callable, x: jnp.ndarray) -> tuple:
    return mean_f(x, theta), cov_f(theta, x, x)



def sample_prior(
        n_functions : int, 
        params      : dict, 
        x           : jnp.ndarray, 
        kernel      : callable, 
        jitter      : float = 1e-6, 
        mean_f      : callable = None, 
        show_plot   : bool = False
    ) -> jnp.ndarray:

    if mean_f is None:
        def zero_mean(x, params):
            return jnp.zeros(x.shape[0])
        mean_f = zero_mean

    n_samples = x.shape[0]

    logging.debug(f"sample_prior:inputs: n_functions={n_functions}, params={params}, x.shape={x.shape}, kernel={kernel}, jitter={jitter}, mean_f={mean_f}, show_plot={show_plot}")

    # random samples from distribution
    x = x[:n_samples, :].copy()

    cov_f = _cov_func(kernel)
    mu_x, cov_x = gp_prior(params, mean_f, cov_f, x)

    # make it semi-positive definite with jitter
    cov_x_ = cov_x + jitter * jnp.eye(cov_x.shape[0])

    # Jax random numbers boilerplate code
    key = jax.random.PRNGKey(0)

    y_samples = jax.random.multivariate_normal(key, mu_x, cov_x_, shape=(n_functions))

    if show_plot:
        plt.figure(figsize=(4, 3))
        plt.title(f"{n_functions} samples from kernel")
        for isample in y_samples:
            plt.plot(isample)
        plt.tight_layout()
        plt.show()
    
    # To use y_samples to plot the sample: y_samples.flatten()
    return y_samples


def cholesky_factorization(K: jnp.ndarray, Y: jnp.ndarray) -> tuple:
    # cho factor the cholesky
    L = jax.scipy.linalg.cho_factor(K, lower=True)
    # weights
    weights = jax.scipy.linalg.cho_solve(L, Y)
    return L, weights


def posterior(
        params           : dict, 
        prior_funcs      : tuple, 
        X                : jnp.ndarray, 
        Y                : jnp.ndarray, 
        Y_err            : jnp.ndarray,
        X_new            : jnp.ndarray, 
        likelihood_noise : bool = False, 
        return_cov       : bool = False,
        jit_key          : str = 'jit'
    ) -> tuple:

    (mean_func, cov_func) = prior_funcs

    # ==========================
    # 1. GP PRIOR
    # ==========================
    _, Kxx = gp_prior(params, mean_f=mean_func, cov_f=cov_func, x=X.squeeze())

    # ===========================
    # 2. CHOLESKY FACTORIZATION
    # ===========================
    logging.debug(f"CHOL: Y: {Y.shape}, {mean_func(X.squeeze(), params).shape}")

    (L, lower), alpha = cholesky_factorization(
        Kxx + (params[jit_key]**2 + Y_err**2) * jnp.eye(Kxx.shape[0]), 
        Y-mean_func(X.squeeze(), params).reshape(-1,1)
    )

    # ================================
    # 4. PREDICTIVE MEAN DISTRIBUTION
    # ================================
    # calculate transform kernel
    KxX = cov_func(params, X_new, X)

    # Calculate the Mean
    mu_y = jnp.dot(KxX, alpha)

    # =====================================
    # 5. PREDICTIVE COVARIANCE DISTRIBUTION
    # =====================================
    v = jax.scipy.linalg.cho_solve((L, lower), KxX.T)
    
    # Calculate kernel matrix for inputs
    Kxx = cov_func(params, X_new, X_new)
    
    cov_y = Kxx - jnp.dot(KxX, v) # "- jnp.dot(KxX, v)" is variance term

    # Likelihood Noise
    if likelihood_noise is True:
        cov_y += params[jit_key]

    # return variance (diagonals of covaraince)
    if return_cov is not True:
        var_y = jnp.diag(cov_y)
        return mu_y, var_y

    return mu_y, cov_y

    
def marginal_likelihood(
        mu_x       : jnp.ndarray, 
        cov_func   : callable, 
        params     : dict, 
        X          : jnp.ndarray, 
        Y          : jnp.ndarray, 
        Y_err      : jnp.ndarray, 
        jitter_key : str = 'jit'
    ) -> float:
     
    # ==========================
    # 1. Covariance Matrix
    # ==========================
    Kxx = cov_func(params, X, X)
    logging.debug(f"Kxx: {Kxx.shape}")

    # ===========================
    # 2. GP Likelihood
    # ===========================
    K_gp = Kxx + (params[jitter_key]**2 + Y_err**2) * jnp.eye(Kxx.shape[0])
    logging.debug(f"K_gp: {K_gp.shape}")

    # ===========================
    # 3. Log Probability
    # ===========================
    log_prob = jax.scipy.stats.multivariate_normal.logpdf(x=Y.T, mean=mu_x, cov=K_gp)

    return jnp.where(jnp.isfinite(log_prob.sum()), log_prob.sum(), -jnp.inf)
