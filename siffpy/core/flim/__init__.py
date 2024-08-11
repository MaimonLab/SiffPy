"""
The `flim` module contains classes and functions for working with
fluorescence lifetime photon arrival time data. It does not deal with
timeseries data explicitly, and is concentrated on photon arrival time histograms.
For timeseries tools or summary statistics like "empirical lifetime" and "phasors",
see the `siffpy.siffmath.flim` module.
"""

from siffpy.core.flim.flimparams import FLIMParams, Exp, Irf # noqa: F401
from siffpy.core.flim.multi_pulse import MultiPulseFLIMParams, MultiIrf, FractionalIrf # noqa: F401
from siffpy.core.flim.flimunits import FlimUnits, convert_flimunits # noqa: F401

import numpy as np

def default_flimparams(n_exps : int = 2) -> FLIMParams:
    """
    Returns a default FLIMParams object with the requested number
    of exponentials.
    """

    taus = np.linspace(0.6, 4.0, n_exps)
    frac = np.ones(n_exps) / n_exps
    irf = Irf(mean = 1.5, sigma = 0.1, units = FlimUnits.NANOSECONDS)

    exps = [Exp(tau = tau, frac = frac, units = FlimUnits.NANOSECONDS) for tau, frac in zip(taus, frac)]

    return FLIMParams(*exps, irf, noise = 0.1)
