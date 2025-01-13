"""
The `flim` module contains classes and functions for working with
fluorescence lifetime photon arrival time data. It does not deal with
timeseries data explicitly, and is concentrated on photon arrival time histograms.
For timeseries tools or summary statistics like "empirical lifetime" and "phasors",
see the `siffpy.siffmath.flim` module.
"""
from typing import Union
from siffpy.core.flim.flimparams import FLIMParams, Exp, Irf # noqa: F401
from siffpy.core.flim.multi_pulse import MultiPulseFLIMParams, MultiIrf, FractionalIrf # noqa: F401
from siffpy.core.flim.flimunits import FlimUnits, convert_flimunits # noqa: F401

import numpy as np

def default_flimparams(n_irfs : int = 1, n_exps : int = 2) -> Union[FLIMParams, MultiPulseFLIMParams]:
    """
    Returns a default FLIMParams object with the requested number
    of exponentials and laser pulses.

    # Arguments

    n_irfs : int
        The number of IRFs to include in the model. If 1, a single
        Gaussian IRF is used. If 2, two Gaussian IRFs are used.

    n_exps : int
        The number of exponential components to include in the model.

    # Returns

    flimparams : FLIMParams
        The default FLIMParams object with the requested number of
        exponentials and laser pulses (will be a MultiPulseFLIMParams
        object if n_irfs > 1).
    """

    taus = np.linspace(0.6, 4.0, n_exps)
    frac = np.ones(n_exps) / n_exps
    exps = [Exp(tau = tau, frac = frac, units = FlimUnits.NANOSECONDS) for tau, frac in zip(taus, frac)]

    if n_irfs == 1:
        irf = Irf(mean = 1.5, sigma = 0.1, units = FlimUnits.NANOSECONDS)

        return FLIMParams(*exps, irf, noise = 0.1)

    else:
        irfs = [
            FractionalIrf(
                mean = 1.5, sigma = 0.1, frac = 0.5, units = FlimUnits.NANOSECONDS,
            ),
            FractionalIrf(
                mean = 4.5, sigma = 0.1, frac = 0.5, units = FlimUnits.NANOSECONDS,
            ),
        ]
        return MultiPulseFLIMParams(
            *exps, *irfs, noise = 0.1
        )
