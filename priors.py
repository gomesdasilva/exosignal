import sys
import numpy as np
import scipy.stats
import matplotlib.pylab as plt


# Available prior names and their corresponding classes:
AV_PRIORS = dict(
    fixed      = "FixedPrior",
    uniform    = "UniformPrior",
    normal     = "NormalPrior",
    loguniform = "LogUniformPrior",
    beta       = "BetaPrior",
    tc_prior   = "TcPrior",         # conditional on keplerian period
    per_cond   = "PerCondPrior",    # conditional on multiples of the QPGP eta3 period
)


# def _convert_dict_to_Priors(priors: dict) -> dict:
#     """Convert priors (dict of lists) to Priors class (dict of classes)."""
#     p = {}
#     for (name, entries) in zip(priors.keys(), priors.values()):
#         assert len(entries) >= 1, f"Priors ERROR: {key}: Priors entries should be lists or tuples with at least one entries, {entries} was given."

#         assert len(name.split("_")) >= 3, f"Priors ERROR: Prior 'name' {name} must have at least three components separated by '_': If model prior: <dataset_id>_<model_id>_<model_parameter_id>_<model_number>. If instrument prior: <dataset_id>_<instr_param_id>_<instr>."

#         func = entries[0]
#         values = entries[1:]
#         if func == 'fixed':
#             p[name] = Priors(name, func, values[0])
#         elif func == 'tc_prior':
#             p[name] = Priors(name, func)
#         else:
#             p[name] = Priors(name, func, *values)
#     return p


class BasePrior:
    _par_names = None
    _scipy_func = None


    def __init__(self, name: str, func: str, pars: list) -> None:
        assert isinstance(pars, (np.ndarray, list, tuple, None)), f"Priors error: {name} | {self.__class__.__name__}: prior parameters must be ndarray, list or tuple."

        if self._par_names:
            assert len(pars) == len(self._par_names), f"Priors error: {name} | {self.__class__.__name__}: distribution requires {len(self._par_names)} parameters, but {len(pars)} were given."

        self.name = name
        self.func = func
        self.pars = pars

        if isinstance(self._scipy_func, scipy.stats._distn_infrastructure.rv_frozen) or isinstance(self._scipy_func, scipy.stats.rv_continuous):
            self._is_scipy_dist = True
        else:
            self._is_scipy_dist = False

        self.__repr__ = self.repr()


    def repr(self) -> str:
        pars_str = ''
        if isinstance(self._par_names, (np.ndarray, list, tuple)):
            for i, (key, val) in enumerate(zip(self._par_names, self.pars)):
                if i == len(self.pars) - 1 and len(self.pars) > 1:
                    pars_str += ", "
                pars_str += f"{key}={val}"
        return f"{self.__class__.__name__}({pars_str})"
    

    def make_dist(self, *kwargs) -> scipy.stats._distn_infrastructure.rv_frozen:
        return self._scipy_func(**self._scipy_pars)
    

    def sample(self, size: int, *kwargs) -> np.ndarray | float:
        """Sample from distribution."""
        samples = self.dist.rvs(size)

        if isinstance(samples, np.ndarray) and samples.size == 1:
            return samples[0]
        return samples
    

    def plot(self, x: np.ndarray, show: bool = True, ax: plt.Axes | None = None, **kwargs) -> None:
        """Plot distribution density."""
        if not ax:
            _, ax = plt.subplots(figsize=(4, 3), layout='constrained')

        ax.plot(x, self.dist.pdf(x), **kwargs)

        if show:
            plt.show()

    
    def logprior(self, theta: float, *kwargs) -> float:
        """Logarithm of the posterior distribution function. To use with MCMC."""
        return self._scipy_func.logpdf(theta, **self._scipy_pars)


    def transform(self, u: float, *kwargs) -> float:
        """Percent point function, to get specific value from the probability value. To use with Nested Sampling."""
        return self._scipy_func.ppf(u, **self._scipy_pars)
    

# ========================================================
# Priors based on Scipy distributions
# ========================================================
class UniformPrior(BasePrior):
    _par_names = ['lo', 'hi']
    _scipy_func = scipy.stats.uniform

    def __init__(self, name: str, func: str, pars: list) -> None:
        super().__init__(name, func, pars)

        # Get prior input parameters
        lo, hi = self.pars

        # Assert prior input parameters
        assert lo < hi, f"Priors error: {self.name} | {self.__class__.__name__}: the first parameter should have a lower value than the second."

        # Make scipy distribution parameters
        self._scipy_pars = dict(loc=lo, scale=hi-lo)

        # Make distribution
        self.dist = self.make_dist()


class NormalPrior(BasePrior):
    _par_names = ['mu', 'sigma']
    _scipy_func = scipy.stats.norm

    def __init__(self, name: str, func: str, pars: list) -> None:
        super().__init__(name, func, pars)

        # Get prior input parameters
        mu, sigma = self.pars

        # Assert prior input parameters
        assert sigma > 0.0, f"Priors error: {self.name}  | {self.__class__.__name__}: the 'sigma' parameter must be positive."

        # Make scipy distribution parameters
        self._scipy_pars = dict(loc=mu, scale=sigma)

        # Make distribution
        self.dist = self.make_dist()


class LogUniformPrior(BasePrior):
    _par_names = ['lo', 'hi']
    _scipy_func = scipy.stats.loguniform

    def __init__(self, name: str, func: str, pars: list) -> None:
        super().__init__(name, func, pars)

        # Get prior input parameters
        lo, hi = self.pars

        # Assert prior input parameters
        assert lo < hi, f"Priors error: {self.name} | {self.__class__.__name__}: the first parameter should have a lower value than the second."

        assert lo > 0.0, f"Priors error: {self.name} | {self.__class__.__name__}: the first parameter must be higher than zero."

        # Make scipy distribution parameters
        self._scipy_pars = dict(a=lo, b=hi)

        # Make distribution
        self.dist = self.make_dist()


class BetaPrior(BasePrior):
    _par_names = ['a', 'b']
    _scipy_func = scipy.stats.beta

    def __init__(self, name: str, func: str, pars: list) -> None:
        super().__init__(name, func, pars)

        # Get prior parameters
        a, b = self.pars

        # Assert prior input parameters
        assert a > 0.0, f"Priors error: {self.name} | {self.__class__.__name__}: the first parameter must be higher than zero."

        assert b > 0.0, f"Priors error: {self.name} | {self.__class__.__name__}: the second parameter must be higher than zero."

        # Make scipy distribution parameters
        self._scipy_pars = dict(a=a, b=b)

        # Make distribution
        self.dist = self.make_dist()


    def plot(self, x: np.ndarray, show: bool = True, ax: plt.Axes | None = None, **kwargs) -> None:
        x = x[x<=1] # defined between 0 and 1.
        super().plot(x, show=show, ax=ax, **kwargs)




# ========================================================
# Fixed and conditional priors
# ========================================================
class FixedPrior(BasePrior):
    _par_names = ['val']

    def __init__(self, name: str, func: str, pars: list) -> None:
        super().__init__(name, func, pars)

        self.dist = self.make_dist()

    def make_dist(self):
        return self.pars[0]

    def sample(self, size: int, *kwargs) -> float:
        return self.pars[0]
    
    def plot(self, x: np.ndarray, show: bool = True, ax: plt.Axes | None = None, **kwargs) -> None:
        """Plot distribution density."""
        if not ax:
            _, ax = plt.subplots(figsize=(4, 3), layout='constrained')

        ax.plot(self.pars[0], 1, **kwargs)

        if show:
            plt.show()
    
    def logprior(self) -> float:
        return 0
    
    def transform(self, u: float, *kwargs) -> float:
        return print("Not implemented.")


class TcPrior(BasePrior):
    """Uniform prior conditional on keplerian period. Finds the best epoch (time of inferior conjuction) closer to half the time span. See Juliet paper for more information."""
    _par_names = None
    _scipy_func = scipy.stats.uniform

    def __init__(self, name: str, func: str, pars) -> None:
        super().__init__(name, func, pars)

        dataset = self.name.split("_")[0]
        num = self.name.split("_")[-1]
        self.per_key = dataset + "_kep_per_" + num


    def sample(self, size: int, t: np.ndarray, priors: dict) -> np.ndarray | float:
        _, per_hi = priors[self.per_key][1:]

        lo = t.mean() - per_hi/2
        hi = t.mean() + per_hi/2
        
        self._scipy_pars = dict(loc=lo, scale=hi-lo)
        self.dist = self._scipy_func(**self._scipy_pars)
        samples = self.dist.rvs(size)

        if isinstance(samples, np.ndarray) and samples.size == 1:
            return samples[0]
        return samples


    def plot(self, x: np.ndarray, priors: dict, show: bool = True, ax: plt.Axes | None = None, **kwargs) -> None:
        """Plot distribution density."""
        if not ax:
            _, ax = plt.subplots(figsize=(4, 3), layout='constrained')

        _, per_hi = priors[self.per_key][1:]

        lo = x.mean() - per_hi/2
        hi = x.mean() + per_hi/2

        ax.plot(x, self._scipy_func(lo, (hi - lo)).pdf(x), **kwargs)

        if show:
            plt.show()


    def logprior(self, theta: float, theta_dict: dict, t: np.ndarray, priors: dict) -> float:
        per_val = theta_dict[self.per_key]

        lo, hi = priors[self.per_key].pars
        if per_val < lo or per_val > hi:
            return -np.inf
        
        lo = t.mean() - per_val/2
        hi = t.mean() + per_val/2

        if theta < lo or theta > hi:
            return -np.inf
        
        self._scipy_pars = dict(loc=lo, scale=hi-lo)

        return self._scipy_func.logpdf(theta, **self._scipy_pars)
    

    def transform(self, u: float, u_dict: dict, t: np.ndarray, priors: dict) -> float:
        u_per = u_dict[self.per_key]

        lo, hi = priors[self.per_key].pars

        if priors[self.per_key].func == 'uniform':
            per_pars = [u_per * lo, u_per * (hi-lo)]
        if priors[self.per_key].func in ['loguniform', 'normal']:
            per_pars = [u_per * lo, u_per * hi]

        x_per = self._scipy_func.ppf(u, *per_pars)

        lo = t.mean() - x_per/2
        hi = t.mean() + x_per/2
        return self._scipy_func.ppf(u, lo, (hi-lo))


class PerCondPrior(BasePrior):
    """Uniform prior whose limits are conditional on multiples of the QPGP eta3 period. Helps limit the eta2 exponential decay timescale to avoid overfitting."""
    _par_names = ['nper_lo', 'nper_hi']
    _scipy_func = scipy.stats.uniform

    def __init__(self, name: str, func: str, pars) -> None:
        super().__init__(name, func, pars)

        dataset = self.name.split("_")[0]
        num = self.name.split("_")[-1]
        self.per_key = dataset + "_qpgp_eta3_" + num


    def sample(self, size: int, priors: dict) -> np.ndarray | float:
        per_lo, per_hi = priors[self.per_key][1:]
        nper_lo, nper_hi = self.pars

        lo =  nper_lo * per_lo
        hi = nper_hi * per_hi
        
        self._scipy_pars = dict(loc=lo, scale=hi-lo)
        self.dist = self._scipy_func(**self._scipy_pars)
        samples = self.dist.rvs(size)

        if isinstance(samples, np.ndarray) and samples.size == 1:
            return samples[0]
        return samples
    

    def plot(self, x: np.ndarray, priors: dict, show: bool = True, ax: plt.Axes | None = None, **kwargs) -> None:
        """Plot distribution density."""
        if not ax:
            _, ax = plt.subplots(figsize=(4, 3), layout='constrained')

        per_lo, per_hi = priors[self.per_key][1:]
        nper_lo, nper_hi = self.pars

        lo =  nper_lo * per_lo
        hi = nper_hi * per_hi

        ax.plot(x, self._scipy_func(lo, (hi - lo)).pdf(x), **kwargs)

        if show:
            plt.show()


    def logprior(self, theta: float, theta_dict: dict, t: np.ndarray, priors: dict) -> float:
        per_val = theta_dict[self.per_key]

        per_lo, per_hi = priors[self.per_key].pars
        nper_lo, nper_hi = self.pars

        lo = nper_lo * per_lo
        hi = nper_hi * per_hi

        if per_val < lo or per_val > hi:
            return -np.inf

        if theta < lo or theta > hi:
            return -np.inf
        
        self._scipy_pars = dict(loc=lo, scale=hi-lo)

        return self._scipy_func.logpdf(theta, **self._scipy_pars)
    

    def transform(self, u: float, u_dict: dict, t: np.ndarray, priors: dict) -> float:
        u_per = u_dict[self.per_key]

        per_lo, per_hi = priors[self.per_key].pars
        nper_lo, nper_hi = self.pars

        if priors[self.per_key].func == 'uniform':
            per_pars = [u_per * per_lo, u_per * (per_hi-per_lo)]
        if priors[self.per_key].func in ['loguniform', 'normal']:
            per_pars = [u_per * per_lo, u_per * per_hi]

        x_per = self._scipy_func.ppf(u, *per_pars)

        lo = nper_lo * x_per
        hi = nper_hi * x_per
        return self._scipy_func.ppf(u, lo, (hi-lo))



class Priors:

    def __init__(self, name: str, func: str, *pars) -> None:
        assert func in AV_PRIORS.keys(), f"Priors error: '{func}' not a defined prior. Available priors are: {list(AV_PRIORS.keys())}."

        self.name = name
        self.func = func
        self.pars = pars

        class_name = AV_PRIORS[func]
        self.prior_class = globals()[class_name](name, func, pars)

        self.__repr__ = self.prior_class.__repr__


    @classmethod
    def convert_dict_to_Priors(cls, priors: dict) -> dict:
        """Convert priors (dict of lists) to Priors class (dict of classes)."""
        p = {}
        for (name, entries) in zip(priors.keys(), priors.values()):
            assert len(entries) >= 1, f"Priors ERROR: {key}: Priors entries should be lists or tuples with at least one entries, {entries} was given."

            assert len(name.split("_")) >= 3, f"Priors ERROR: Prior 'name' {name} must have at least three components separated by '_': If model prior: <dataset_id>_<model_id>_<model_parameter_id>_<model_number>. If instrument prior: <dataset_id>_<instr_param_id>_<instr>."

            func = entries[0]
            values = entries[1:]
            if func == 'fixed':
                p[name] = cls(name, func, values[0])
            elif func == 'tc_prior':
                p[name] = cls(name, func)
            else:
                p[name] = cls(name, func, *values)
        return p


    def make_dist(self, *kwargs) -> object:
        return self.prior_class.make_dist(*kwargs)
    

    def sample(self, size: int, *kwargs) -> np.ndarray | float:
        return self.prior_class.sample(size, *kwargs)
    

    def plot(self, x: np.ndarray, **kwargs) -> None:
        self.prior_class.plot(x, **kwargs)
    

    def logprior(self, theta: float, *kwargs) -> float:
        return self.prior_class.logprior(theta, *kwargs)
    

    def prior_transform(self, u: float, *kwargs) -> float:
        return self.prior_class.transform(u, *kwargs)




# Testing:
if __name__ == '__main__':

    # Test Priors class
    priors_dict = dict(
        y1_kep_per_1 = ['uniform', 1, 100],
        y1_kep_k_1   = ['uniform', 0, 10],
        y1_kep_ecc_1 = ['beta', 0.87, 3.03],
        y1_kep_w_1   = ['uniform', 0, 2*np.pi],
        y1_kep_tc_1  = ['tc_prior'],
        #y1_kep_tc_1  = ['wrong_name', 0, 10],

        y1_qpgp_eta1_1 = ['loguniform', 0.1, 100],
        y1_qpgp_eta2_1 = ['per_cond', 0.5, 5],
        y1_qpgp_eta3_1 = ['uniform', 1, 200],
        y1_qpgp_eta4_1 = ['loguniform', 0.1, 3],
    )

    priors = _convert_dict_to_Priors(priors_dict)
    print(priors)

    # Test per_cond prior:
    priors['y1_qpgp_eta2_1'].plot(x=np.linspace(0, 200, 50000), priors=priors_dict)


    #sys.exit()

    print(priors['y1_kep_per_1'].func)

    for key in priors_dict.keys():
        print(key, "\t:", priors[key].__repr__)

    # TcPrior tests:
    x = np.linspace(0, 500, 50000)
    print(priors['y1_kep_tc_1'].sample(1, t=x, priors=priors_dict))
    priors['y1_kep_tc_1'].plot(x, priors=priors_dict)

    print(priors['y1_kep_per_1'].sample(5))

    print(priors['y1_kep_ecc_1'].sample(3))
    priors['y1_kep_ecc_1'].plot(x)


    sys.exit()

    # Test individual prior classes:
    x = np.linspace(0, 20, 50000)

    prior = UniformPrior("test", "uniform", [5, 15])

    samples = prior.sample(50000)

    _, ax = plt.subplots(figsize=(4, 3), layout='constrained')

    prior.plot(x, ax=ax, show=False)
    ax.hist(samples, bins=40, density=True)
    ax.set_title(prior.__repr__)
    ax.set_ylabel("Distribution density")
    ax.set_xlabel("$x$")
    plt.show()

    prior = LogUniformPrior("test", "normal", [5, 15])
    samples = prior.sample(50000)

    _, ax = plt.subplots(figsize=(4, 3), layout='constrained')

    prior.plot(x, ax=ax, show=False)
    ax.hist(samples, bins=40, density=True)
    ax.set_title(prior.__repr__)
    ax.set_ylabel("Distribution density")
    ax.set_xlabel("$x$")
    plt.show()

    prior = NormalPrior("test", "normal", [10, 2])
    samples = prior.sample(50000)

    _, ax = plt.subplots(figsize=(4, 3), layout='constrained')

    prior.plot(x, ax=ax, show=False)
    ax.hist(samples, bins=40, density=True)
    ax.set_title(prior.__repr__)
    ax.set_ylabel("Distribution density")
    ax.set_xlabel("$x$")
    plt.show()

    prior = BetaPrior("test", "beta", [0.87, 3.03])
    samples = prior.sample(50000)

    _, ax = plt.subplots(figsize=(4, 3), layout='constrained')

    #x_plot = x[x<=1] # Beta function limited to 0 < x < 1

    prior.plot(x, ax=ax, show=False)
    ax.hist(samples, bins=40, density=True)
    ax.set_title(prior.__repr__)
    ax.set_ylabel("Distribution density")
    ax.set_xlabel("$x$")
    plt.show()

    prior = FixedPrior("test", "fixed", [10])
    samples = prior.sample(50000)

    _, ax = plt.subplots(figsize=(4, 3), layout='constrained')

    prior.plot(x, ax=ax, show=False)
    ax.hist(samples, bins=40, density=True)
    ax.set_title(prior.__repr__)
    ax.set_ylabel("Distribution density")
    ax.set_xlabel("$x$")
    plt.show()



    # test assertion errors:
    try:
        priors = UniformPrior("test_error", "uniform", [5])
        print("Assertion error: Failed!")
    except AssertionError as msg:
        print("Debugging:", msg)
        print("Assertion error: OK")
    try:
        priors = UniformPrior("test_error", "uniform", [10, 5])
        print("Assertion error: Failed!")
    except AssertionError as msg:
        print("Debugging:", msg)
        print("Assertion error: OK")