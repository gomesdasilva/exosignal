"""
ExoSignal: a framework for exoplanet radial-velocity signal detection and characterization using Gaussian Processes to model stellar activity and dynamic nested sampling of the posteriors.
Author: João Gomes da Silva

This module contains the main class ExoSignal, which takes a dataframe and a dictionary of priors as input and creates mean models and kernels for each dataset and instrument based on the prior keys. It also contains methods for computing the log-likelihood, running nested sampling, and saving/loading results.
"""
from datetime import datetime
import os
import numpy as np
import pandas as pd
import pickle

from functools import partial
import jax
import jax.numpy as jnp

from priors import Priors
import models
import gp_model
import kernels
import sampling

from signal_plot import SignalPlot



class ExoSignal:
    __version__ = "0.1"

    def __init__(self, 
            df          : pd.DataFrame,
            priors      : dict,
            datasets    : list[str] = ['rv', 'I_CaII'],
            use_offsets : list[bool] = [True, False],
            ref_instr   : str = 'HARPS',  # instrument to use when not using offsets
            key_t       : str = 'bjd',    # column name for time in the dataframe
            key_instr   : str = 'instr',  # column name for instrument in the dataframe
            key_err     : str = '_err',   # suffix to be added to the dataset name to get the error column in the dataframe
            verb        : bool = True
        ) -> None:

        self.df = df
        self.key_t = key_t
        self.key_instr = key_instr
        self.key_err = key_err

        # Assert that the required columns are in the dataframe
        required_keys = datasets + [key_t, key_instr] + [dataset + key_err for dataset in datasets]
        for key in required_keys:
            if key not in df.columns:
                raise ValueError(f"Column '{key}' not found in dataframe. Please check the column names and try again.")

        self.datasets = {"y" + str(num+1): dataset for num, dataset in enumerate(datasets)}
        
        self.instr_ds = self.instr_per_dataset(use_offsets, ref_instr)

        self.instrs_ds = {key: list(np.unique(val)) for key, val in self.instr_ds.items()}


        # Priors:
        priors.update(self.instr_priors(priors))
        self.read_priors(priors)

        # Models to make per dataset
        self.models_ds = self._models_per_dataset()

        self.dataset_model_classes()

        self.print_config(verb=verb)


    def print_config(self, verb=True):
        if verb:
            print()
            print("=======================================================")
            print(f"ExoSignal v{self.__version__} configuration")
            print("=======================================================")
            print("Data")
            print("-------------------------------------------------------")
            print(f"Nobs\t\t= {self.df[self.key_t].shape[0]}")

            # make string with datasets
            dataset_str = ''
            for i, (key, dataset) in enumerate(self.datasets.items()):
                dataset_str += f"{key}: "
                dataset_str += f"{dataset}"
                if i < len(self.datasets)-1:
                    dataset_str += "\n\t\t= "
            print(f"Datasets\t= {dataset_str}")

            # make string with instruments for each dataset
            instr_str = ''
            for j, (dataset_id, instr) in enumerate(self.instr_ds.items()):
                instr_str += f"{dataset_id}: "

                for i, inst in enumerate(np.unique(instr)):
                    instr_str += f"{inst}"
                    if i < len(np.unique(instr))-1:
                        instr_str += ", "

                if j < len(self.instr_ds)-1:
                    instr_str += "\n\t\t= "
            print(f"Instruments\t= {instr_str}")

            print("-------------------------------------------------------")
            print("Models")
            print("-------------------------------------------------------")

            # Deterministic models and parameters:
            models_str = ''
            for j, (dataset_id, mods) in enumerate(self.mean_model_classes_ds.items()):
                models_str += f"{dataset_id}: "

                for i, name in enumerate(mods[0].get_model_names()):
                    models_str += f"{name}"
                    if i < len(mods[0].get_model_names())-1:
                        models_str += ", "

                if j < len(self.mean_model_classes_ds)-1:
                    models_str += "\n\t\t= "
            print(f"Det. Models\t= {models_str}")

            params_str = ''
            for j, (dataset_id, mods) in enumerate(self.mean_model_classes_ds.items()):
                params_str += f"{dataset_id}: "
                for mod in mods:
                    for i, name in enumerate(mod.get_model_keys()):
                        if "_mu_" in name:
                            params_str += f"{name}"
                            if i < len(mod.get_model_names())-1:
                                params_str += ", "
                for i, keys in enumerate(mods[0].get_model_keys()):
                    if "_mu_" not in keys:
                        params_str += f"{keys}"
                        if i < len(mods[0].get_model_keys())-1:
                            params_str += ", "

                if j < len(self.mean_model_classes_ds)-1:
                    params_str += "\n\t\t= "
            print(f"Det. Parameters\t= {params_str}")

            # GP kernels and parameters:
            gp_str = ''
            for j, (dataset_id, kerns) in enumerate(self.kernel_classes_ds.items()):
                if len(kerns[0].kernel_classes) > 0:
                    gp_str += f"{dataset_id}: "

                    for i, name in enumerate(kerns[0].get_kernel_names()):
                        gp_str += f"{name}"
                        if i < len(kerns[0].get_kernel_names())-1:
                            gp_str += ", "

                    if j < len(self.kernel_classes_ds)-1:
                        gp_str += "\n\t\t= "
            if len(kerns[0].kernel_classes) > 0:
                print(f"GP kernels\t= {gp_str}")

            gp_par_str = ''
            for j, (dataset_id, kerns) in enumerate(self.kernel_classes_ds.items()):
                if len(kerns[0].kernel_classes) > 0:
                    gp_par_str += f"{dataset_id}: "

                    for i, keys in enumerate(kerns[0].get_kernel_keys()):
                        gp_par_str += f"{keys}"
                        if i < len(kerns[0].get_kernel_keys())-1:
                            gp_par_str += ", "

                    if j < len(self.kernel_classes_ds)-1:
                        gp_par_str += "\n\t\t= "
            
            if len(kerns[0].kernel_classes) > 0:
                print(f"GP parameters\t= {gp_par_str}")

            print("-------------------------------------------------------")
            print("Priors")
            print("-------------------------------------------------------")
            for key, prior in self.priors.items():
                print(f"{key}\t= {prior.__repr__}")


    def read_priors(self, priors: dict) -> None:
        """Read priors and select variable and fixed variables."""
        # Convert priors dictionary to Priors class (dictionary of priors classes)
        priors = Priors.convert_dict_to_Priors(priors)

        # Get keys for variables
        vary_keys = [key for key in priors if priors[key].func != 'fixed']

        # Make dictionary with fixed parameters
        fixed_params = {key: priors[key].pars[0] for key in priors if priors[key].func == 'fixed'}

        self.priors       = priors
        self.vary_keys    = vary_keys
        self.fixed_params = fixed_params


    def instr_per_dataset(self, use_offsets: list[bool], ref_instr: str) -> dict:
        instr = {key: self.df.instr.values for key in self.datasets.keys()}

        # make new instr if not using offset for a given dataset
        for (dataset_id, dataset), use_offset in zip(self.datasets.items(), use_offsets):
            if use_offset is False:
                instr[dataset_id] = np.array([ref_instr]*self.df[dataset].size)

        return instr


    def instr_priors(self, priors: dict) -> dict:
        """Create mean and jitter priors for each dataset and for each instrument (if not given as input)."""
        priors_instr = dict()

        for dataset_id in self.datasets:
            y     = self.df[self.datasets[dataset_id]].values
            y_err = self.df[self.datasets[dataset_id] + "_err"].values
            y_min = y.min() #- 3*y.std()
            y_max = y.max() #+ 3*y.std()

            for instr in self.instrs_ds[dataset_id]:
                if dataset_id + "_mu_" + instr not in priors:
                    mu_prior = dataset_id + '_mu_' + instr
                    priors_instr[mu_prior] = ['uniform', y_min, y_max]

                if dataset_id + "_jit_" + instr not in priors:
                    priors_instr[dataset_id + '_jit_' + instr] = ['loguniform', np.min(y_err)*1e-1, np.max(y) - np.min(y)]

        return priors_instr


    def _models_per_dataset(self) -> dict:
        """Decide which models to make based on the priors names.
        
        This function depends on a fixed prior name convention where:
        priors name = <dataset_id>_<model_prior_id>_<model_param_id>_<model_num>
        """
        # initialize models_dataset dictionary
        models_dataset = {}
        for dataset_id in self.datasets:
            models_dataset[dataset_id] = []

        # make list of model to make and number for each dataset
        for key in self.priors.keys():
            if "_mu_" not in key and "_jit_" not in key:
                prior_id = key.split("_")[-3]
                model_num = key.split("_")[-1]
                for dataset_id in self.datasets:
                    if dataset_id in key and [prior_id, model_num] not in models_dataset[dataset_id]:
                        models_dataset[dataset_id].append([prior_id, model_num])

        return models_dataset


    def make_model_dataset(self, dataset_id: str) -> tuple[list[models.MeanModel], list[kernels.Kernels]]:
        """Make models for each instrument of a given dataset."""


        def make_models(model, dataset_id: str) -> models.MeanModel | kernels.Kernels:
            """Connects the prior ids with the model parameters from Models and Kernel classes."""
            if isinstance(model, models.MeanModel):
                class_name = models
                av_models = model.av_models # Available deterministic models for mean function
                add_model = model.add_model # Method to add a new model to the mean function

            elif isinstance(model, kernels.Kernels):
                class_name = kernels
                av_models = model.av_kernels # Available kernels for GPs
                add_model = model.add_kernel # Method to add a new kernel to the GP model

            for model_id, model_n in self.models_ds[dataset_id]:
                for av_model, av_id in av_models.T:
                    # check if model_id is in the available models
                    if model_id == av_id:
                        # initialise model
                        new_model = getattr(class_name, av_model)()
                        # attribute prior keys to model
                        for i, key in enumerate(new_model.keys):
                            new_key = model_id + "_" + key + "_" + model_n
                            if dataset_id + "_" + new_key in self.priors:
                                new_model.keys[i] = dataset_id + "_" + new_key
                            elif "sh_" + new_key in self.priors:
                                new_model.keys[i] = "sh_" + new_key

                        add_model(new_model)
            return model


        mean_model_list = []
        kernel_list = []
        for instr in self.instrs_ds[dataset_id]:
            mean_model = models.MeanModel()
            mean_model.model_classes[0].keys = [dataset_id+'_mu_'+instr]

            kernel = kernels.Kernels()

            mean_model = make_models(mean_model, dataset_id)
            kernel = make_models(kernel, dataset_id)

            mean_model_list.append(mean_model)
            kernel_list.append(kernel)

        assert len(mean_model_list) == len(self.instrs_ds[dataset_id])

        return mean_model_list, kernel_list


    def dataset_model_classes(self) -> None:
        """Make a dictionary for mean model and kernel for each dataset."""
        self.mean_model_classes_ds = dict()
        self.kernel_classes_ds = dict()

        for dataset_id in self.datasets:
            mean_model_list, kernel_list = self.make_model_dataset(dataset_id)
            self.mean_model_classes_ds[dataset_id] = mean_model_list
            self.kernel_classes_ds[dataset_id] = kernel_list



    def mean_models(self, dataset_id: str, theta: dict) -> list[jnp.ndarray]:
        """Mean (deterministic) model for a given dataset, which can have multiple instruments. This is used as input for the GP log-likelihood calculation."""
        y_model = []
        for i, instr in enumerate(self.instrs_ds[dataset_id]):
            mask = self.instr_ds[dataset_id] == instr

            x = np.array(self.df[mask][self.key_t].values)

            model = self.mean_model_classes_ds[dataset_id][i]

            y_model.append(model.compute(x, theta))
        return y_model
    
    

    @partial(jax.jit, static_argnums=(0, 2))
    def log_likelihood_dataset(self, theta: dict, dataset_id: str, mean_models: list[jnp.ndarray]) -> float:
        """Log-likelihood for a given dataset, which can have multiple instruments."""
        theta.update(self.fixed_params)
        loglike = 0
        for i, instr in enumerate(self.instrs_ds[dataset_id]):
            mask = self.instr_ds[dataset_id]==instr

            x = jnp.vstack(jnp.array(self.df[mask][self.key_t].values))
            y = jnp.vstack(jnp.array(self.df[mask][self.datasets[dataset_id]].values))
            y_err = jnp.vstack(jnp.array(self.df[mask][self.datasets[dataset_id] + '_err'].values))

            mean_model = mean_models[i]

            kernel = self.kernel_classes_ds[dataset_id][i]

            if not kernel.kernel_classes:
                kernel = kernels.ConstKernel()

            cov_f = gp_model._cov_func(kernel.compute)

            loglike += gp_model.marginal_likelihood(mean_model, cov_f, theta, x, y, y_err, jitter_key=dataset_id + '_jit_' + instr)

        return loglike



    # For markov chain monte carlo sampling:
    def log_likelihood(self, theta: dict) -> float:
        """Log-likelihood for all datasets. Can be used by MCMC samplers like emcee."""
        theta.update(self.fixed_params)
        loglike = 0
        for dataset_id in self.datasets:
            mean_model = self.mean_models(dataset_id, theta)
            loglike += self.log_likelihood_dataset(theta, dataset_id, mean_model)
        return loglike


    # For nested sampling:
    def log_likelihood_nested(self, params: list[float]) -> float:
        """Log-likelihood for all datasets to be used by Dynesty or other nested samplers. Here the parameters input is list of values."""
        theta = {key: p for (key, p) in zip(self.vary_keys, params)}
        return self.log_likelihood(theta)

    # For nested sampling:
    def prior_transform(self, u: list[float]) -> list[float]:
        """Transform prior space to unit cubes for nested sampling."""
        prior_list = []

        # make u_dict to be used by conditional priors
        u_dict = {key: u_i for u_i, key in zip(u, self.priors)}
        for i, key in enumerate(self.vary_keys):
            prior_list.append(self.priors[key].prior_transform(u[i], u_dict, self.df[self.key_t].values, self.priors))

        return jnp.array(prior_list)


    def run_dynesty(self, nlive: int = 500, walks_c: int = 3, save_path: str | None = None, verb: bool = True):
        self.save_path = save_path
        self.nlive = nlive

        if self.save_path is not None:
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

        LL = self.log_likelihood_nested
        gradLL = jax.jit(jax.grad(LL))

        prior_transf = self.prior_transform # function to transform priors to unit cubes
        ndim = len(self.vary_keys)
        walks = int(ndim * walks_c)

        start = datetime.now()
        sampler, samples = sampling.run_dynesty(LL, gradLL, prior_transf, ndim, nlive, walks, save_path=save_path, verb=verb)
        end = datetime.now()
        self.duration = end - start

        lnZ = sampler.results['logz'][-1]
        lnZ_err = sampler.results['logzerr'][-1]
        if verb is True:
            print("=======================================================")
            print("Results")
            print("-------------------------------------------------------")
            print(f"Runtime\t\t= {self.duration}")
            print(f"lnZ\t\t= {lnZ:.2f} +/- {lnZ_err:.2f}")

        posteriors = {key: val for key, val in zip(self.vary_keys, samples)}

        if save_path is not None:
            save_file = os.path.join(save_path, "posteriors.csv")
            pd.DataFrame(posteriors).to_csv(save_file, index=False)

        post_med = {key: np.median(post) for key, post in posteriors.items()}
        post_med.update(self.fixed_params)

        if verb is True:
            print("-------------------------------------------------------")
            print("Median posteriors")
            print("-------------------------------------------------------")
            for key, posterior in posteriors.items():
                med = np.median(posterior)
                hi = np.quantile(posterior, 0.64) - med
                lo = med - np.quantile(posterior, 0.16)
                print(f"{key}\t= {med:.4f} (+{hi:.4f} -{lo:.4f})")
            print("=======================================================")

        self.posteriors = posteriors
        self.post_med   = post_med
        self.lnZ        = lnZ
        self.lnZ_err    = lnZ_err


    def save_results(self) -> dict:
        self._save_logs()

        results = dict()
        results['lnZ']            = self.lnZ
        results['lnZ_err']        = self.lnZ_err
        results['posteriors']     = self.posteriors
        results['post_med']       = self.post_med
        results['mean_models_ds'] = self.mean_model_classes_ds
        results['kernels_ds']     = self.kernel_classes_ds
        results['instr_ds']       = self.instr_ds
        results['datasets']       = self.datasets
        results['df']             = self.df
        results['key_t']          = self.key_t
        results['key_instr']      = self.key_instr
        results['key_err']        = self.key_err

        save_file = os.path.join(self.save_path, "results.pickle")
        pickle.dump(results, open(save_file, 'wb'))
        return results


    def _save_logs(self) -> None:
        with open(os.path.join(self.save_path, "config.txt"), "w") as f:
            # Config:
            print(f"n_obs = {self.df.shape[0]}", file=f)
            print(f"ndim = {len(self.posteriors)}", file=f)
            print(f"nlive = {self.nlive}", file=f)
            print(f"datasets = {self.datasets}", file=f)
            for dataset_id, instr in self.instr_ds.items():
                print(f"instrs: {dataset_id}: {np.unique(instr)}", file=f)
            for dataset_id, model in self.mean_model_classes_ds.items():
                print(f"models: {dataset_id}: {model[0].get_model_names()}", file=f)
            for dataset_id, kernel in self.kernel_classes_ds.items():
                print(f"kernels: {dataset_id}: {kernel[0].get_kernel_names()}", file=f)
            print("", file=f)
            print("Priors:", file=f)
            for key, prior in self.priors.items():
                print(f"{key}: {prior.func}{prior.pars}", file=f)

        with open(os.path.join(self.save_path, "results.txt"), "w") as f:
            # Results:
            #print("", file=f)
            print("Date =", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file=f)
            print(f"Runtime = {self.duration}", file=f)
            print(f"lnZ = {self.lnZ:.1f} ± {self.lnZ_err:.1f}", file=f)
            print("", file=f)
            print("Posteriors:", file=f)
            for key, posterior in self.posteriors.items():
                med = np.median(posterior)
                print(f"{key} = {med:.4f} + {(np.quantile(posterior, 0.84) - med):.4f} - {(med - np.quantile(posterior, 0.16)):.4f}", file=f)


    @staticmethod
    def load_results(load_path: str) -> dict:
        load_file = os.path.join(load_path, "results.pickle")
        results = pickle.load(open(load_file, 'rb'))
        return results



    @staticmethod
    def corner_plot(posteriors: dict, save_path: str | None = None, show: bool = True) -> None:
        """Make a corner plot of the posteriors."""
        sampling.corner_plot(posteriors, save_path=save_path, show=show)


    @staticmethod
    def plot(results: dict, save_path: str | None = None) -> SignalPlot:
        """Plotting class for the results."""
        return SignalPlot(results, save_path)