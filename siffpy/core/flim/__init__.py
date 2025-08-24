"""
The `flim` module contains classes and functions for working with
fluorescence lifetime photon arrival time data. It does not deal with
timeseries data explicitly, and is concentrated on photon arrival time histograms.
For timeseries tools or summary statistics like "empirical lifetime" and "phasors",
see the `siffpy.siffmath.flim` module.
"""
from typing import Union
from copy import deepcopy
from siffpy.core.flim.flimparams import FLIMParams, Exp, Irf # noqa: F401
from siffpy.core.flim.multi_pulse import MultiPulseFLIMParams, MultiIrf, FractionalIrf # noqa: F401
from siffpy.core.flim.multi_color import MPMFMCFlimParams # noqa: F401
from siffpy.core.flim.flimunits import FlimUnits, convert_flimunits, FlimUnitsLike # noqa: F401

import numpy as np

def default_flimparams(
        n_irfs : int = 1,
        n_exps : int = 2,
        n_fluorophores : int = 1,
        n_channels : int = 1
    ) -> Union[FLIMParams, MultiPulseFLIMParams]:
    """
    Returns a default FLIMParams object with the requested number
    of exponentials and laser pulses. Note that this creates a
    set of homogenous `FLIMParams` objects when there are multiple
    fluorophores (i.e. they have the same `Exp`s and `Irf`s at
    initialization). For `MPMFMCFlimParams` objects, the `chi`s
    are not identical.

    # Arguments

    n_irfs : int
        The number of IRFs to include in the model. If 1, a single
        Gaussian IRF is used. If 2, two Gaussian IRFs are used.

    n_exps : int
        The number of exponential components to include in the model.

    n_fluorophores : int
        The number of fluorophores producing the data

    n_channels : int
        The number of detection channels
        
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
        irf = [Irf(mean = 1.5, sigma = 0.1, units = FlimUnits.NANOSECONDS)]
        if n_fluorophores == 1:
            return FLIMParams(*exps, *irf, noise = 0.1)
    else:
        irf = [
            FractionalIrf(
                mean = 1.5 + 3*n, sigma = 0.05, frac = 1/n_irfs, units = FlimUnits.NANOSECONDS,
            )
            for n in range(n_irfs)
        ]

    if n_fluorophores == 1:
        return MultiPulseFLIMParams(
            *exps, *irf, noise = 0.1
        )
    

    if n_channels == 1:
        raise NotImplementedError("Haven't tested multi-fluorophore one channel FPs yet")

    fluorophores = [
        MultiPulseFLIMParams(
            *deepcopy(exps), *deepcopy(irf), noise = 0.0
        )
        for f_idx in range(n_fluorophores)
    ] 

    chis = [
        np.random.dirichlet([0.5]*n_channels)
    ] * n_fluorophores

    return MPMFMCFlimParams(
        *fluorophores,
        n_colors = n_channels,
        chis = chis
    )    

