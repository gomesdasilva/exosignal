"""
Adapted from RadVel (Fulton+ 2014)
"""
import numpy as np


def rv_drive(t: np.ndarray, k: float, per: float, ecc: float, tp: float, w: float) -> np.ndarray:
    """RV Drive.

    Parameters:
    -----------
        t : (array)
            times of observations [days]
        k : float
            RV semi-amplitude [m/s]
        per : float
            Orbital period [days]
        ecc : float
            Eccentricity
        tp : float
            Time of periastron passage [days]
        w : float
            Argument of periastron [radians]

    Returns:
    --------
        rv: (array of floats): radial velocity model
    """

    # Performance boost for circular orbits
    if ecc == 0.0:
        m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
        return k * np.cos(m + w)

    if per < 0:
        per = 1e-4
    if ecc < 0:
        ecc = 0
    if ecc > 0.99:
        ecc = 0.99

    # Calculate the approximate eccentric anomaly, E1, via the mean anomaly  M.
    nu = true_anomaly(t, tp, per, ecc)
    rv = k * (np.cos(nu + w) + ecc * np.cos(w))

    return rv


def kepler(Marr: np.ndarray, eccarr: np.ndarray) -> np.ndarray:
    """Solve Kepler's Equation.

    Parameters:
    -----------
        Marr : array
            Input Mean anomaly
        eccarr : array
            Eccentricity

    Returns:
    --------
        array: eccentric anomaly
    """

    # convergence criterion
    conv = 1.0e-12
    k = 0.85

    # first guess at E
    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr
    # fiarr should go to zero when converges
    fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)
    # which indices have not converged
    convd = np.where(np.abs(fiarr) > conv)[0]
    # number of unconverged elements
    nd = len(convd)
    count = 0

    # while unconverged elements exist
    while nd > 0:
        count += 1

        # just the unconverged elements ...
        M = Marr[convd]
        ecc = eccarr[convd]
        E = Earr[convd]

        # fi = E - e*np.sin(E)-M    ; should go to 0
        fi = fiarr[convd]
        # d/dE(fi) ;i.e.,  fi^(prime)
        fip = 1 - ecc * np.cos(E)
        # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fipp = ecc * np.sin(E)
        # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)
        fippp = 1 - fip

        # first, second, and third order corrections to E
        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        E = E + d3
        Earr[convd] = E
        # how well did we do?
        fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr)
        # test for convergence
        convd = np.abs(fiarr) > conv
        nd = np.sum(convd is True)

    if Earr.size > 1:
        return Earr
    else:
        return Earr[0]




def true_anomaly(t: np.ndarray, tp: float, per: float, ecc: float) -> np.ndarray:
    """
    Calculate the true anomaly for a given time, period, eccentricity.

    Parameters:
    -----------
        t : array
            Array of times [days]
        tp : float
            Time of periastron [days]
        per : float
            Orbital period in days [days]
        ecc : float
            Eccentricity

    Returns:
    --------
        array: true anomoly at each time
    """

    # f in Murray and Dermott p. 27
    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
    eccarr = np.zeros(t.size) + ecc
    e1 = kepler(m, eccarr)
    n1 = 1.0 + ecc
    n2 = 1.0 - ecc
    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(e1 / 2.0))

    return nu


def timetrans_to_timeperi(tc: float, per: float, ecc: float, w: float) -> float:
    """
    Convert Time of Transit to Time of Periastron Passage.

    Parameters:
    -----------
        tc : float
            time of transit [days]
        per : float
            period [days]
        ecc : float
            eccentricity
        w : float
            longitude of periastron [radians]

    Returns:
    --------
        float: time of periastron passage

    """
    try:
        if ecc >= 1:
            return tc
    except ValueError:
        pass

    f = np.pi/2 - w
    # eccentric anomaly
    ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))
    # time of periastron
    tp = tc - per/(2*np.pi) * (ee - ecc*np.sin(ee))

    return tp


def timeperi_to_timetrans(tp: float, per: float, ecc: float, w: float, secondary: bool = False) -> float:
    """
    Convert Time of Periastron to Time of Conjuction.

    Parameters:
    -----------
    tp : float
        Time of periastron [days]
    per : float
        Period [days]
    ecc : float
        Eccentricity
    w : float
        Argument of peri [radians]
    secondary : bool, optional
        Calculate time of secondary eclipse (time of superior conjunction) instead (default is False)


    Returns:
    --------
        float: time of inferior conjunction (time of transit if system is transiting)

    """
    try:
        if ecc >= 1:
            return tp
    except ValueError:
        pass


    if secondary:
        # true anomaly during secondary eclipse
        f = 3*np.pi/2 - w
        # eccentric anomaly
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))

        # ensure that ee is between 0 and 2*pi (always the eclipse AFTER tp)
        if isinstance(ee, np.float64):
            ee = ee + 2 * np.pi
        else:
            ee[ee < 0.0] = ee + 2 * np.pi
    else:
        # true anomaly during transit
        f = np.pi/2 - w
        # eccentric anomaly
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))

    # time of conjunction (transit or secondary eclipse)
    tc = tp + per/(2*np.pi) * (ee - ecc*np.sin(ee))

    return tc
