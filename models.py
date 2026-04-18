import os
import numpy as np 
import jax.numpy as jnp

import importlib
from inspect import isclass

import keplerian
from keplerian import rv_drive



class BaseModel:
    """Base model class.
    
    All models should inherit from this class.
    """

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


class ConstModel(BaseModel):
    """Constant model.
    
    This model is used automatically to calculate the mean value of the data.
    """

    _prior_id = 'const'

    def __init__(self) -> None:
        self._keys = ['mu']
        self._name = self.__class__.__name__

    def compute(self, x: jnp.ndarray, theta: dict) -> jnp.ndarray:
        return jnp.zeros(x.shape[0]) + theta[self.keys[0]]


class TrendModel(BaseModel):
    """Linear trend model."""

    _prior_id = 'trend'

    def __init__(self) -> None:
        self._keys = ['slope']
        self._name = self.__class__.__name__

    def compute(self, x: jnp.ndarray, theta: dict) -> jnp.ndarray:
        return theta[self.keys[0]] * (x - jnp.min(x))


class QuadModel(BaseModel):
    """Quadratic model."""

    _prior_id = 'quad'

    def __init__(self) -> None:
        self._keys = ['a']
        self._name = self.__class__.__name__

    def compute(self, x: jnp.ndarray, theta: dict) -> jnp.ndarray:
        return theta[self.keys[0]] * (x - np.min(x))**2


class SinModel(BaseModel):
    """Sinusoidal model."""

    _prior_id = 'sin'

    def __init__(self) -> None:
        self._keys = ['amp', 'per', 'phi']
        self._name = self.__class__.__name__

    def compute(self, x: jnp.ndarray, theta: dict) -> jnp.ndarray:
        return theta[self.keys[0]] * jnp.sin(2*jnp.pi*x/theta[self.keys[1]] + theta[self.keys[2]])


class KepModel(BaseModel):
    """Keplerian RV model."""

    _prior_id = 'kep'

    def __init__(self) -> None:
        self._keys = ['k', 'per', 'ecc', 'tc', 'w']
        self._name = self.__class__.__name__

    def compute(self, x: jnp.ndarray, theta: dict) -> jnp.ndarray:
        k, per, e, tc, w = theta[self._keys[0]], theta[self._keys[1]], theta[self._keys[2]], theta[self._keys[3]], theta[self._keys[4]]

        tp = keplerian.timetrans_to_timeperi(tc, per, e, w)
        return rv_drive(x, k, per, e, tp, w)
    



class MeanModel:
    """Class to manage the mean model of the data, which is the sum of all the models added to it.
    """

    __this_module = importlib.import_module(os.path.basename(__file__).split(".")[0])
    

    def __init__(self) -> None:
        self.model_classes = [ConstModel()]

        __ignore_modules = [self.__class__.__name__, BaseModel.__name__]

        __av_model_names = [name for name in dir(self.__this_module) if isclass(getattr(self.__this_module, name)) if name not in __ignore_modules]

        __av_model_prior_id = [getattr(self.__this_module, name)._prior_id for name in __av_model_names]

        self.av_models = np.array([__av_model_names, __av_model_prior_id])


    def get_model_names(self) -> list:
        names = [model.name for model in self.model_classes]
        return names
        

    def get_model_keys(self) -> list:
        keys = [model.keys for model in self.model_classes]
        return np.concatenate(keys)


    def get_model_prior_ids(self) -> list:
        prior_ids = [model.prior_ids for model in self.model_classes]
        return prior_ids


    def add_model(self, model_class: BaseModel) -> None:
        # important note: model_class must be initialized
        assert not isinstance(model_class, type), "MeanModel Error: model_class must be initialized first"
        self.model_classes.append(model_class)
    

    def compute(self, t: np.ndarray, theta: dict) -> np.ndarray:
        return sum((model.compute(t, theta) for model in self.model_classes))


    def fit(self, t: np.ndarray, y: np.ndarray) -> None:
        """Fit models to data using e.g. curve_fit"""
        raise NotImplemented


# testing:
if __name__ == '__main__':
    model = MeanModel()
    print(model.av_models)

    model.add_model(KepModel())

    theta = dict(mu=0, k=5, per=20, ecc=0.5, tc=0, w=5)
    x = np.linspace(0, 100, 100)
    y = model.compute(x, theta)

    import matplotlib.pylab as plt
    plt.plot(x, y, 'k.-')
    plt.show()





