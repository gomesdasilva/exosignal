import numpy as np
import matplotlib
import matplotlib.pylab as plt

from typing import Optional, Literal, Tuple

try:
    from astropy.timeseries import LombScargle
except:
    try:
        from astropy.stats import LombScargle
    except:
        print("WARNING: astropy periodogram not available.")

from scipy.signal import find_peaks




class Periodogram:

    @staticmethod
    def gls(
            t      : list | np.ndarray,
            y      : list | np.ndarray,
            y_e    : Optional[list | np.ndarray] = None,
            pmin   : Optional[float] = 2,
            pmax   : Optional[int] = None,
            method : Literal['astropy'] = 'astropy',
        ) -> dict:
        """Calculates GLS periodogram using several algorithms."""
        if method == 'astropy':
            gls = Periodogram._gls_astropy(t, y, y_e, pmin, pmax)

        return gls


    @staticmethod
    def _gls_astropy(
            t     : list | np.ndarray,
            y     : list | np.ndarray,
            y_e   : Optional[list | np.ndarray] = None,
            pmin  : Optional[float] = 1.5,
            pmax  : Optional[float | int] = None,
            steps : Optional[int] = None,
        ) -> dict:
        """Calculates GLS periodogram using astropy algorithm."""
        t = np.array(t)
        y = np.array(y)
        y_e = np.array(y_e)

        tspan = np.ptp(t)

        if not pmax:
            pmax = tspan

        cadence = np.diff(np.sort(t))
        cadence = cadence[cadence != 0.]
        cadence_min = cadence.min()

        oversampling_factor = 2
        nyquist_p = cadence_min / (2 * oversampling_factor)

        steps = pmax/nyquist_p

        fmin = 1./pmax
        fmax = 1./pmin

        freq = np.linspace(fmin, fmax, int(steps))
        period = 1./freq

        if isinstance(y_e, bool) and y_e is None:
            gls = LombScargle(t, y)
        else:
            gls = LombScargle(t, y, y_e)

        power = gls.power(freq)

        # analytical FAPs using astropy's method, which is based on Baluev 2008
        faps = gls.false_alarm_probability(power)

        # Different FAP levels can be chosen here
        fap_levels = [0.1, 0.01, 0.001]
        fap10, fap1, fap01 = gls.false_alarm_level(fap_levels)

        # window function
        y_win = np.ones_like(y)
        power_win = LombScargle(t, y_win, fit_mean=False, center_data=False).power(freq)

        gls = dict(
            period    = period,
            power     = power,
            power_win = power_win,
            faps      = faps,
            fap10     = fap10,
            fap1      = fap1,
            fap01     = fap01,
        )

        return gls
    

    @staticmethod
    def _find_peaks(
            gls : dict
        ) -> Tuple[np.ndarray[float]]:
        """Retrieve all the peaks in the periodogram."""
        peaks, _ = find_peaks(gls['power'])

        if len(peaks) == 0:
            print(f"No peaks found in GLS: peaks = {peaks}.")
            return
        
        peaks_power, peaks_period, peaks_fap = zip(*sorted(zip(gls['power'][peaks], gls['period'][peaks], gls['faps'][peaks])))

        peaks_power  = peaks_power[::-1]
        peaks_period = peaks_period[::-1]
        peaks_fap    = peaks_fap[::-1]

        return peaks_power, peaks_period, peaks_fap
    

    @staticmethod
    def _mark_fap_lvl(
            gls  : dict,
            ax   : matplotlib.axes._axes.Axes,
            faps : Optional[list[int | float]] = [10, 0.1]
        ) -> None:
        """Mark the FAP levels in a plot as horizontal lines and labels.

        Parameters:
        -----------
            gls : dict
                The results from calculating the GLS
            ax : matplotlib.axes._axes.Axes
                matpltolib axes
            faps : (list, optional)
                List of FAP values in %. Defaults to [10, 0.1].
        """
        for fap in faps:
            fap_str = str(fap)

            if "." in fap_str:
                fap_str = str(fap).replace(".", "")
            else:
                fap_str = fap

            ax.axhline(gls[f'fap{fap_str}'], color='grey', ls=':', lw=0.7)
            ax.text(gls['period'].max(), gls[f'fap{fap_str}'], f"{fap}%", fontsize=7, horizontalalignment='right')


    @staticmethod
    def signifcant_periods(
            gls       : dict,
            fap_limit : Optional[int | float] = 10
        ) -> Tuple[list[float]]:
        """Retrieve the significant periods in the periodogram.

        Parameters:
        -----------
            gls : dict
                Dictionary with results from calculating the periodogram.
            fap_limit : (int, float)
                Minimum FAP below which the GLS peaks are considered ssignificant.

        Returns:
        --------
            sign_periods : list
                List of significant periods.
            sign_faps : list
                List of the FAPs of each significant period.
        """
        _, peaks_period, peaks_fap = Periodogram._find_peaks(gls)

        fap_limit /= 100 # to fraction

        sign_periods = []
        sign_faps = []
        for period, fap in zip(peaks_period, peaks_fap):
            if fap <= fap_limit:
                sign_periods.append(period)
                sign_faps.append(fap)

        return sign_periods, sign_faps
    

    @staticmethod
    def plot(
            t           : list | np.ndarray,
            y           : list | np.ndarray,
            y_e         : Optional[list | np.ndarray] = None,
            pmin        : Optional[float | int] = 1.5,
            pmax        : Optional[float | int] = None,
            ax_in       : Optional[matplotlib.axes._axes.Axes] = None,
            method      : Literal['astropy'] = 'astropy',
            title       : Optional[str] = '',
            period_in   : Optional[float | int] = None,
            figsize     : Optional[Tuple[int | float]] = (4, 2),
            label       : Optional[str] = '',
            maxper_show : Optional[int] = 3,
        ) -> None:
        """Plot periodogram."""
        if not pmax:
            pmax = np.ptp(t)

        if method=='astropy':
            gls = Periodogram._gls_astropy(t, y, y_e, pmin=pmin, pmax=pmax)

        if ax_in is None:
            _, ax = plt.subplots(figsize=figsize, layout='constrained')
        else:
            ax = ax_in

        ax.set_title(title)

        ax.semilogx(gls['period'], gls['power_win'], c='lightgrey', ls='-', lw=0.7)
        ax.semilogx(gls['period'], gls['power'], 'k-', lw=0.7)

        ax.annotate(label, xy=(0.95, 0.85), xycoords='axes fraction', horizontalalignment='right')

        Periodogram._mark_fap_lvl(gls, ax, faps=[10, 1, 0.1])

        sign_periods, _ = Periodogram.signifcant_periods(gls)

        for i, period in enumerate(sign_periods[:maxper_show]):
            ax.axvline(period, ls=':', color='b')
            str = rf"$P_{i+1}$ = " + f"{period:.2f} d"
            ax.annotate(str, xy=(0.05, 0.85 - i*0.1), xycoords='axes fraction', fontsize=8)

        if period_in:
            ax.axvline(period_in, color='g', ls='-', alpha=0.5)

        ax.set(ylabel='GLS power', xlabel="Period [d]")

        ax.minorticks_on()

        if ax_in is None:
            plt.show()



# Testing:
if __name__ == "__main__":

    np.random.seed(123)
    t = np.linspace(900, 1000, 5000)
    t = np.random.choice(t, 100)
    t = np.sort(t)

    y0 = np.sin(2*np.pi*t/23.5) + np.sin(2*np.pi*t/40 + 0.3)

    sigma = 0.3
    noise = sigma * (1 + np.random.normal(0, 0.2, 100))
    y_e = noise

    y = y0 + np.random.normal(0, noise, 100)
    
    plt.figure()
    plt.errorbar(t, y, y_e, fmt='k.')
    plt.plot(t, y0)
    plt.show()

    Periodogram.plot(t, y, y_e, pmin=1, period_in=30, label=r'H$\alpha$', maxper_show=2)