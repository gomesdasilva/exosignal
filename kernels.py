import numpy as np 
import jax
import jax.numpy as jnp
import functools

import importlib
from inspect import isclass

import gp_model


# Squared Euclidean Distance Formula
@jax.jit
def sqeuclidean_distance(x: jnp.array, y: jnp.array) -> float:
    return jnp.sum((x-y)**2)

# Euclidean Distance Formula
@jax.jit
def euclidean_distance(x: jnp.array, y: jnp.array) -> float:
    return jnp.sum(jnp.sqrt((x-y)**2))



class BaseKernel:
    """Base kernel class."""

    _prior_id = ''

    def __init__(self) -> None:
        self._keys = []
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @property
    def keys(self) -> list:
        return self._keys

    @keys.setter
    def keys(self, new_keys: list) -> None:
        self._keys = new_keys  



class ConstKernel(BaseKernel):
    """Constant kernel.
    
    This kernel is used automatically when no GP is used, i.e., it is the "null" kernel, which adds no covariance to the model.
    """

    _prior_id = 'constgp'

    def __init__(self) -> None:
        self._keys = []
        self._name = self.__class__.__name__

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute(self, theta: dict, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return 0.0



class SEKernel(BaseKernel):
    """Squared Exponential Kernel."""

    _prior_id = 'segp'

    def __init__(self) -> None:
        self._keys = ['eta1', 'eta2']
        self._name = self.__class__.__name__

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute(self, theta: dict, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return theta[self._keys[0]]**2 * jnp.exp(-sqeuclidean_distance(x, y)/theta[self._keys[1]]**2)



class ES2Kernel(BaseKernel):
    """Exponential Sine Squared Kernel."""

    _prior_id = 'es2gp'

    def __init__(self) -> None:
        self._keys = ['eta1', 'eta3', 'eta4']
        self._name = self.__class__.__name__

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute(self, theta: dict, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return theta[self._keys[0]]**2 * jnp.exp(-jnp.sin(jnp.pi*euclidean_distance(x, y)/theta[self._keys[1]])**2 /(2.*theta[self._keys[2]]**2))



class QPKernel(BaseKernel):
    """Quasi-Periodic Kernel."""

    _prior_id = 'qpgp'

    def __init__(self) -> None:
        self._keys = ['eta1', 'eta2', 'eta3', 'eta4']
        self._name = self.__class__.__name__

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute(self, theta: dict, x: jnp.ndarray, y: jnp.ndarray) -> float:
        SE = jnp.exp(-sqeuclidean_distance(x, y)/theta[self._keys[1]]**2)
        ES2 = jnp.exp(-jnp.sin(jnp.pi*euclidean_distance(x, y)/theta[self._keys[2]])**2 /(2.*theta[self._keys[3]]**2))
        return theta[self._keys[0]]**2 * SE * ES2



class Kernels:
    """Class to manage GP kernels."""

    __this_module = importlib.import_module("kernels")
    
    def __init__(self) -> None:
        self.kernel_classes = []

        __ignore_modules = [self.__class__.__name__, BaseKernel.__name__]

        __av_kernel_names = [name for name in dir(self.__this_module) if isclass(getattr(self.__this_module, name)) if name not in __ignore_modules]

        __av_kernel_prior_id = [getattr(self.__this_module, name)._prior_id for name in __av_kernel_names]

        self.av_kernels = np.array([__av_kernel_names, __av_kernel_prior_id])

    def get_kernel_names(self) -> list:
        names = [kernel.name for kernel in self.kernel_classes]
        return names
        
    def get_kernel_keys(self) -> np.ndarray:
        keys = [kernel.keys for kernel in self.kernel_classes]
        return np.concatenate(keys)

    def get_kernel_prior_ids(self) -> list:
        prior_ids = [kernel.prior_ids for kernel in self.kernel_classes]
        return prior_ids

    def add_kernel(self, kernel_class: BaseKernel) -> None:
        # important note: model_class must be initialized first
        assert not isinstance(kernel_class, type), "kernel_class must be initialized first"
        self.kernel_classes.append(kernel_class)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute(self, theta: dict, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return sum((kernel.compute(theta, x, y) for kernel in self.kernel_classes))



# testing:
if __name__ == '__main__':
    kernels = Kernels()
    print(kernels.av_kernels)


    theta = dict(
        eta1 = np.sqrt(3),
        eta2 = 150,
        eta3 = 100,
        eta4 = 1.0,
        jit = 0.1
    )

    t = np.linspace(0, 500, 30)

    import gp_model

    cov1 = gp_model.cov_matrix(QPKernel().compute, theta, t, t)
    print(cov1)

    cov1_f = gp_model._cov_func(QPKernel().compute)

    import models
    y = models.SinModel().compute(t, dict(amp=5, per=40, phi=0))
    y_err = 0.1 * np.ones_like(y)


    ll_1 = gp_model.marginal_likelihood(np.zeros_like(y), cov1_f, theta, t[:, None], y[:, None], y_err[:, None], jitter_key='jit')
    print(ll_1)

    kernels.add_kernel(QPKernel())

    gp_model.sample_prior(3, theta, t[:, None], QPKernel().compute, show_plot=True)


