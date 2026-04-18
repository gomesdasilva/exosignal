import functools
import numpy as np
import matplotlib.pylab as plt
import jax
import jax.numpy as jnp
#from jax.experimental import optimizers
import tqdm

import os
import dynesty
from dynesty import utils as dyfunc
import pickle
import corner

# There is a warning about overflow in the log-likelihood calculation, but it does not (seem to) affect the results.
import warnings
warnings.filterwarnings("ignore")



def run_dynesty(
        loglike         : callable, 
        grad_log_like   : callable, 
        prior_transform : callable, 
        ndim            : int, 
        nlive           : int, 
        walks           : int, 
        save_path       : str | None = None, 
        verb            : bool = True
    ) -> tuple[dynesty.DynamicNestedSampler, np.ndarray]:
    """Run dynesty dynamic nested sampling.
    
    Notes:
    1. Working with gaussian processes:
       sample = 'rwalk' is better when sampling from GP
       In low dimensions (ndim < 15) use walks < 25. For higher dimensions (ndim ~15-25) use walks = 50 or higher. E.g. when fitting a QPGP only for 50 points, walks = 10 is very fast (3:43 min) [walks = 5 is faster ;)].
    2. For simple deterministic functions, use sample = 'auto' or 'rwalk' with
       walks = 5.
    3. High walks values (e.g. 25) makes the sampling very slow. Made a test using walks=5 and walks=25 for same model (2QPGP, N=300), and in the first case it took 6 min to run, while in the second it took ~32 min, with the same lnZ and similar results!
    """
    sample = 'rwalk' # well-suited for handling correlated parameter distributions, with an increased number of MCMC walk steps to enhance convergence.

    if verb is True:
        print("-------------------------------------------------------")
        print(" Dynesty configuration")
        print("-------------------------------------------------------")
        print(f"Ndim\t\t= {ndim}")
        print(f"Nlive\t\t= {nlive}")
        print(f"Sample\t\t= {sample}")
        print(f"Walks\t\t= {walks}")
        print("-------------------------------------------------------")
        print("Running dynesty...")
        print("-------------------------------------------------------")
    

    sampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim, sample=sample, gradient=grad_log_like, walks=walks, bound="multi")
    sampler.run_nested(nlive_init=nlive)

    if save_path is not None:
        file_path = os.path.join(save_path, 'dynesty_results.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(sampler.results, file)

    samples = sampler.results.samples  # samples
    weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])  # normalized weights
    ess = np.sum(weights)**2 / np.sum(weights**2)
    print("ESS:", ess) # Effective Sample Size should be higher than nlive. Values > 10000 for high-precision work.

    samples = dyfunc.resample_equal(samples, weights)

    return sampler, samples.T



def corner_plot(posteriors: dict, save_path: str | None = None, show: bool = True) -> None:
    """Make a corner plot of the posteriors."""

    samples = np.array(list(posteriors.values()))

    fig = plt.figure(figsize=(len(posteriors)*1, len(posteriors)*1))

    corner.corner(samples.T,
        labels=list(posteriors.keys()),
        label_kwargs={"fontsize": 6},
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 6},
        plot_datapoints=False,
        fig=fig,
        lw=0.7,
        smooth=True,
        color='k',
    )

    for ax in fig.axes:
        ax.tick_params(axis='both', which='major', labelsize=6)

    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "corner_plot.pdf"))

    if show:
        plt.show()
