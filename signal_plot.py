import os
import numpy as np
import matplotlib.pylab as plt

import jax.numpy as jnp

import gp_model
import models
import kernels
from periodogram import Periodogram


class SignalPlot:

    def __init__(self, results: dict, save_path: str | None = None) -> None:
        self.df             = results['df']
        self.key_t          = results['key_t']
        self.key_err        = results['key_err']
        self.save_path      = save_path
        self.posteriors     = results['posteriors']
        self.post_med       = results['post_med']
        self.mean_models_ds = results['mean_models_ds']
        self.kernels_ds     = results['kernels_ds']
        self.datasets       = results['datasets']
        self.instr_ds       = results['instr_ds']


    def _make_full_model_dataset(self, dataset_id: str) -> dict:
        y_center     = []
        full_model   = []
        full_model_i = []
        ti_i         = []
        residuals    = []
        for i, instr in enumerate(np.unique(self.instr_ds[dataset_id])):
            
            mask = self.instr_ds[dataset_id]==instr

            t     = self.df[self.key_t][mask].values[:, None]
            y     = self.df[self.datasets[dataset_id]].values[mask][:, None]
            y_err = self.df[self.datasets[dataset_id] + self.key_err].values[mask][:, None]

            mu = self.post_med[dataset_id + '_mu_' + instr]

            # deterministic model
            mean_func = self.mean_models_ds[dataset_id][i].compute

            ti = np.linspace(t.squeeze().min(), t.squeeze().max(), 1000)[:, None] # time grid for plotting models

            # GP model
            kernel = self.kernels_ds[dataset_id][i]
            cov_func = gp_model._cov_func(kernel.compute)
            gp_priors = (mean_func, cov_func)

            mean_y, _   = gp_model.posterior(self.post_med, gp_priors, t, y, y_err, t, False, False, jit_key=dataset_id + '_jit_' + instr)

            mean_y_i, _ = gp_model.posterior(self.post_med, gp_priors, t, y, y_err, ti, False, False, jit_key=dataset_id + '_jit_' + instr)

            # full model
            y_model = mean_y.squeeze() + mean_func(t.squeeze(), self.post_med) - mu
            y_model_i = mean_y_i.squeeze() + mean_func(ti.squeeze(), self.post_med) - mu # for plotting models
            # residuals
            y_res = y.squeeze() - y_model - mu

            y_center.append(y.squeeze() - mu)
            full_model.append(y_model)
            full_model_i.append(y_model_i)
            residuals.append(y_res)
            ti_i.append(ti)

        model = dict()
        model['y_center']     = np.concatenate(y_center)     # Y values subtracted by the mean for each instrument
        model['full_model']   = np.concatenate(full_model)   # Full model evaluated at the data points
        model['residuals']    = np.concatenate(residuals)    # Residuals (Y - Full model)
        model['full_model_i'] = np.concatenate(full_model_i) # For plotting models
        model['ti_i']         = np.concatenate(ti_i)         # Time stamp for plotting models

        return model


    def _plot_full_model_dataset(self, 
            dataset_id : str, 
            save_plot  : bool = False, 
            show_gp    : bool = True, 
            x_label    : str = "BJD [d]", 
            y_label    : str | None = None, 
            y_units    : str = '[m/s]'
        ) -> None:
        """Plot full model for a given dataset."""
        
        model = self._make_full_model_dataset(dataset_id)
        residuals = model['residuals']

        _, ax = plt.subplots(3, 1, figsize=(5, 3*2), gridspec_kw=dict(height_ratios=[2, 1, 1]), sharex=False)

        for i, instr in enumerate(np.unique(self.instr_ds[dataset_id])):
            mask = self.instr_ds[dataset_id]==instr

            t     = self.df[self.key_t][mask].values[:, None]
            y     = self.df[self.datasets[dataset_id]][mask].values[:, None]
            y_err = self.df[self.datasets[dataset_id] + self.key_err][mask].values[:, None]

            mu = self.post_med[dataset_id + "_mu_" + instr]

            ax[0].errorbar(t, (y - mu).squeeze(), y_err.squeeze(), label=instr, marker='o', ls='', capsize=0, ecolor='grey', mec='k', lw=0.7, ms=4, alpha=0.7)

            ax[0].annotate(f"std_{instr} = {np.std(y.squeeze()):.4f}", xy=(0.05, 0.9 - i*0.05), xycoords="axes fraction", fontsize=8)

            ax[1].errorbar(t, residuals[mask], y_err.squeeze(), label=f'std = {residuals[mask].std():.4f}', marker='o', ls='', capsize=0, ecolor='grey', mec='k', lw=0.7, ms=4, alpha=0.7)

            # x grid for plotting
            ti_j = jnp.linspace(t.squeeze().min(), t.squeeze().max(), 10000)[:, None]
            ti = np.linspace(t.squeeze().min(), t.squeeze().max(), 10000)[:, None]
            
            # deterministic model
            mean_func = self.mean_models_ds[dataset_id][i].compute

            # GP model
            kernel = self.kernels_ds[dataset_id][i]
            cov_func = gp_model._cov_func(kernel.compute)
            gp_priors = (mean_func, cov_func)

            # mean and variance for plotting
            mean_yi, var_yi = gp_model.posterior(self.post_med, gp_priors, t, y, y_err, ti_j, likelihood_noise=False, return_cov=False, jit_key=dataset_id + '_jit_' + instr)

            mean_model_i = mean_func(ti.squeeze(), self.post_med) - mu

            uncertainty = 1.96 * jnp.sqrt(var_yi.squeeze()) # 95% confidence level

            if i == 0:
                label_fm = 'Full model'
                label_gp = 'GP model'
            else:
                label_fm = None
                label_gp = None

            # Full predictive mean:
            ax[0].plot(ti, mean_yi.squeeze() + mean_model_i, lw=1, color='k', label=label_fm)
            # GP predictive mean: 
            if kernel is not kernels.ConstKernel and show_gp is True:
                ax[0].plot(ti, mean_yi.squeeze(), color='r', linewidth=1, label=label_gp)

            # Predictive Std (95% Conf.):
            ax[0].fill_between(
                ti.squeeze(),
                mean_yi.squeeze() + mean_model_i + uncertainty,
                mean_yi.squeeze() + mean_model_i - uncertainty, 
                alpha=0.7,
                color='lightgrey',
                )

        ax[0].axhline(0.0, color='k', ls=':', lw=0.7)
        ax[0].legend(fontsize=8, loc=1)

        ax[1].annotate(f"std = {np.std(residuals):.4f}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8)
        ax[1].axhline(0.0, color='k', ls=':', lw=0.7)
        ax[1].legend(fontsize=8, loc=1)
        ax[1].set_xlabel(x_label)

        if y_label is not None:
            ax[0].set_ylabel(y_label + " " + y_units)
        else:
            ax[0].set_ylabel(self.datasets[dataset_id].upper() + " " + y_units)
        ax[1].set_ylabel(r"O$-$C" + " " + y_units)


        # Residuals GLS:
        t = self.df[self.key_t].values
        y_err = self.df[self.datasets[dataset_id] + self.key_err].values


        gls = Periodogram.gls(t, residuals, y_err, pmin=1.5, pmax=np.ptp(t))

        ax[2].semilogx(gls['period'], gls['power'], 'k-', lw=0.7)
        ax[2].semilogx(gls['period'], gls['power_win'], color='grey', ls='-', lw=0.7, alpha=0.7) # Window function periodogram
        ax[2].axhline(gls['fap01'], ls='--', lw=0.7, color='k')
        ax[2].axhline(gls['fap1'], ls=':', lw=0.7, color='k')
        ax[2].set_xlabel('Period [d]')
        ax[2].set_ylabel("Norm. Power")

        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(self.save_path, f"{self.datasets[dataset_id]}_full_model.pdf"))
        plt.show()           


    def _plot_phase_dataset(self, 
            dataset_id  : str, 
            save_plot   : bool = False, 
            y_label     : str  = "$y$", 
            y_units     : str  = '', 
            x_label     : str  = '$\phi$', 
            show_binned : bool = True
        ) -> None:
        """Plot phase folded data for a given dataset."""

        model = self._make_full_model_dataset(dataset_id)
        full_model = model['full_model']
        residuals = model['residuals']

        mean_model = self.mean_models_ds[dataset_id][0] # for the same dataset, if more than one instruments are available, the models used will be the same, hence the first one is chosen.

        for model in mean_model.model_classes:
            # Phase plots only available for periodic deterministic models.
            if isinstance(model, (models.KepModel, models.SinModel)):
                _, ax = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[2, 1]), figsize=(5, 4), sharex=True)

                x_lst     = []
                y_lst     = []
                y_e_lst   = []
                y_res_lst = []
                for instr in np.unique(self.instr_ds[dataset_id]):
                    mask = self.instr_ds[dataset_id] == instr

                    x     = self.df[self.key_t][mask].values
                    y     = self.df[self.datasets[dataset_id]][mask].values
                    y_err = self.df[self.datasets[dataset_id] + self.key_err][mask].values

                    full_model_instr = full_model[mask]

                    mu = self.post_med[dataset_id + "_mu_" + instr]
                    y_center = y - mu

                    per_key = [key for key in model.keys if "_per_" in key][0]
                    # Get periodic function type and number of the periodic component (in case more than one is present in the model) from the period key name:
                    func = per_key.split("_")[-3]
                    num = per_key.split("_")[-1]

                    periodic_func = model.compute
                    y_model = periodic_func(x, self.post_med)
                    y_model_res = y_center - (full_model_instr - y_model)

                    # Phase plot:
                    per = self.post_med[per_key]
                    
                    k_key = [key for key in model.keys if "_k_" in key or "_amp_" in key][0]
                    k = self.post_med[k_key]

                    if func == 'sin':
                        ecc = 0.0
                    elif func == 'kep':
                        ecc_fixed = True
                        ecc = self.post_med[[key for key in model.keys if "_ecc_" in key][0]]
                        for key in self.posteriors.keys():
                            if "_ecc_" in key:
                                ecc_fixed = False

                    if not ecc_fixed: 
                        ecc_err = np.std(self.posteriors[[key for key in model.keys if "_ecc_" in key][0]])
                    
                    per_post = self.posteriors[per_key]
                    per_err = np.mean([np.quantile(per_post, 0.84) - np.median(per_post), np.median(per_post) - np.quantile(per_post, 0.16)])
                    
                    k_post = self.posteriors[k_key]

                    x_lst.append(x)
                    y_lst.append(y_model_res)
                    y_e_lst.append(y_err)
                    y_res_lst.append(y_model_res-y_model)
            
                    if ecc_fixed:
                        ax[0].set_title(f"P = {per:.3f} ± {per_err:.3f} | K = {k:.2f} ± {np.std(k_post):.2f} | ecc = {ecc:.2f} (fixed)", fontsize=9)
                    else:
                        ax[0].set_title(f"P = {per:.3f} ± {per_err:.3f} | K = {k:.2f} ± {np.std(k_post):.2f} | ecc = {ecc:.2f} ± {ecc_err:.2f}", fontsize=9)

                    ax[0].errorbar((x)/per % 1, y_model_res, y_err, marker='o', ls='', label=instr, capsize=0, ecolor='grey', mec='k', lw=0.7, ms=4, alpha=0.7)

                    ax[0].legend(fontsize=8)

                    ax[1].errorbar((x)/per % 1, y_model_res - y_model, y_err, marker='o', ls='', capsize=0, ecolor='grey', mec='k', lw=0.7, ms=4, alpha=0.7)

                if show_binned:
                    x     = np.concatenate(x_lst)/per % 1
                    y     = np.concatenate(y_lst)
                    y_err = np.concatenate(y_e_lst)
                    y_res = np.concatenate(y_res_lst)

                    x, y, y_err, y_res = zip(*sorted(zip(x, y, y_err, y_res)))

                    x_bin, y_bin, y_e_bin = self.func_grid(x, y, y_err, func=np.mean, grid_step=0.1, grid_min=0.0, grid_max=1.0, err='SEM')

                    _, y_res_bin, _ = self.func_grid(x, y_res, y_err, func=np.mean, grid_step=0.1, grid_min=0.0, grid_max=1.0, err='SEM')

                    ax[0].errorbar(x_bin, y_bin, y_e_bin, color='white', ecolor='k', marker='o', mec='k', mew=1, ms=7, ls='', zorder=5, capsize=0)

                    ax[1].errorbar(x_bin, y_res_bin, y_e_bin, color='white', ecolor='k', marker='o', mec='k', mew=1, ms=7, ls='', zorder=5, capsize=0)

                # Plot model in phase space:
                phase = np.linspace(0, per, 1000)
                y_model_phase = periodic_func(phase, self.post_med)
                ax[0].plot(phase/per, y_model_phase, 'k-')

                ax[0].axhline(0.0, color='k', ls=':', lw=0.7)
                if y_label is not None:
                    ax[0].set_ylabel(y_label + " " + y_units)
                else:
                    ax[0].set_ylabel(self.datasets[dataset_id].upper() + " " + y_units)

                ax[1].axhline(0.0, color='k', ls=':', lw=0.7)
                ax[1].annotate(f"std = {np.std(residuals):f}", xy=(0.05, 0.85), xycoords='axes fraction', fontsize=8)
                ax[1].set_xlabel(x_label)
                ax[1].set_ylabel(r"O$-$C" + " " + y_units)

                ax[0].set_xlim(0, 1)
                ax[1].set_xlim(0, 1)

                plt.tight_layout()
                if save_plot:
                    plt.savefig(os.path.join(self.save_path, f"{self.datasets[dataset_id]}_{func}_P{num}_phase.pdf"))
                plt.show()


    def full_model(self,
            save_plot   : bool = False,
            show_gp     : bool = False,
            y_units     : str | None = None,
            y_labels    : str | None = None,
            x_label     : str | None = None
        ) -> None:
        """Plot full model for all datasets."""

        if y_units is not None:
            assert len(self.datasets) == len(y_units), "Number of entries in y_units should be the same as the number of datasets used."
        else:
            y_units = ['']*len(self.datasets)

        if y_labels is not None:
            assert len(self.datasets) == len(y_labels), "Number of entries in y_labels should be the same as the number of datasets used."
        else:
            y_labels = ['']*len(self.datasets)

        for i, dataset_id in enumerate(self.datasets):
            self._plot_full_model_dataset(dataset_id, y_units=y_units[i], y_label=y_labels[i], x_label=x_label, save_plot=save_plot, show_gp=show_gp)


    def phase(self,
            save_plot: bool = False,
            y_units: str | None = None,
            y_labels: str | None = None,
            x_label: str = 'Phase, $\phi$',
            show_binned: bool = True
        ) -> None:
        """Plot phase-folded data for all datasets."""

        if y_units is not None:
            assert len(self.datasets) == len(y_units), "Number of entries in y_units should be the same as the number of datasets used."
        else:
            y_units = ['']*len(self.datasets)

        if y_labels is not None:
            assert len(self.datasets) == len(y_labels), "Number of entries in y_labels should be the same as the number of datasets used."
        else:
            y_labels = ['']*len(self.datasets)

        for i, dataset_id in enumerate(self.datasets):
            self._plot_phase_dataset(dataset_id, y_units=y_units[i], y_label=y_labels[i], x_label=x_label, save_plot=save_plot, show_binned=show_binned)


    @staticmethod
    def func_grid(
            x             : np.ndarray,
            y             : np.ndarray,
            y_e           : np.ndarray,
            grid_min      : float,
            grid_max      : float,
            grid_step     : float = 0.1,
            func          : callable = np.mean,
            quantile      : float = 0.05,
            err           : str = 'sigma',
            show_plot     : bool = False,
            list_of_lists : bool = False
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply a function to y using a 1d grid with a window of size grid_step.
        
        Parameters:
        -----------
            x : array
                Data X coordinate.
            y : array
                Data Y coordinate.
            y_e : array
                Data Y coordinate uncertainty.
            grid_min : float
                Minimum x coordinate of the grid.
            grid_max : float
                Maximum x coordinate of the grid.
            grid_step : float
                Step of the grid.
            func : callable (optional)
                Statistical function to be applied to each step of the grid. Default is 'np.mean'.
            quantile : float (optional)
                Value of the quantile if 'func' is 'np.quantile'.
            err : str (optional)
                The method used to calculate the uncertainties inside each 'grid_step'. Default is 'sigma'. Options are 'sigma' for the standard deviation and 'SEM' for the standard error on the mean.
            show_plot : bool (optional)
                Show diagnostic plot. Default is 'False'.
            list_of_lists : bool (optional)
                If 'True' returns the list of lists including the data points segregated into windows with size 'geid_step'. Default is 'False'.
            
        Returns:
        --------
            x_grid : array
                X coordinate of bin.
            y_grid : array
                Y coordinate of 'func' applyed to y bin.
            y_grid_e : array
                Error of y coordinate of 'func' applyed to y bin.
            x_list_of_lists : array
                x coordinate array with epochs sub-arrays
            y_list_of_lists : array
                y coordinate array with epochs sub-arrays
        """

        # There are some warnings due to empty bins (which are expected), but it does not affect the results.
        import warnings
        warnings.filterwarnings("ignore")

        grid = np.arange(grid_min, grid_max, grid_step)

        func_grid = []
        x_grid = []
        y_list_of_lists = []
        y_e_list_of_lists = []
        x_list_of_lists = []
        for i, xi in enumerate(grid):
            y_list = []
            y_e_list = []
            x_list = []
            for xx, yy, yy_e in zip(x, y, y_e):
                if xx >= xi and xx < xi + grid_step:
                    y_list.append(yy)
                    y_e_list.append(yy_e)
                    x_list.append(xx)
            if func == np.quantile:
                func_grid.append(func(y_list, quantile))
            else:
                func_grid.append(func(y_list))
            x_grid.append(np.mean(x_list))

            x_list_of_lists.append(x_list)
            y_list_of_lists.append(y_list)
            y_e_list_of_lists.append(y_e_list)

        y_grid = np.asarray(func_grid)
        x_grid = np.asarray(x_grid)
        x_list_of_lists = np.array(x_list_of_lists, dtype=object)
        y_list_of_lists = np.array(y_list_of_lists, dtype=object)
        y_e_list_of_lists = np.array(y_e_list_of_lists, dtype=object)

        if show_plot:
            plt.title(f"grid step = {grid_step}", fontsize=10)
            plt.errorbar(x, y, y_e, fmt='k.')
            plt.plot(x_grid, func_grid, marker='o', mfc='none', mec='red', ls='')
            for grid_i in grid:
                plt.axvline(grid_i, color='k', ls=':', lw=0.7)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.show()

        if err == 'std':
            y_grid_e = np.array([np.std(y) for y in y_list_of_lists])
        elif err == 'SEM':
            y_grid_e = np.array([np.std(y)/np.sqrt(len(y)) for y in y_list_of_lists])
        else:
            raise ValueError(f"ERROR: 'err' must be 'std' or 'SEM', but {err} was given")

        if not list_of_lists:
            return x_grid, y_grid, y_grid_e
        elif list_of_lists:
            return x_grid, y_grid, y_grid_e, x_list_of_lists, y_list_of_lists, y_e_list_of_lists