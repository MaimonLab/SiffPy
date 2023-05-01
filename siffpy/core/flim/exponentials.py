"""
Math related to exponentials
used for computing values in
FLIM analysis

Functions
---------

chi_sq = px_chi_sq_exp(photon_arrivals, params)
    Returns the chi-squared value of a data array with the given params.

LL = px_log_likelihood_exp(photon_arrivals, params)
    returns the log-likelihood value of the dataset photon_arrivals
    under the parameter dictionary params

p = monoexponential_probability(photon_arrivals, tau, tau_g)
    Computes the probability of a photon_arrivals vector under the
    assumption of an exponential distribution with time constant tau
    and IRF gaussian width tau_g


SCT March 27 2021
"""

import numpy as np
from scipy.stats import exponnorm

def param_tuple_to_pdf(x_axis : np.ndarray, param_tuple : tuple)->np.ndarray:
    """
    Convert a tuple of parameters to a probability density function
    """
    pdist = np.zeros(x_axis.shape)
    irf_mean, irf_sigma = param_tuple[-2], param_tuple[-1]
    for tau, frac in zip(param_tuple[:-2:2], param_tuple[1:-2:2]):
        pdist += frac * exponnorm.pdf(x_axis, tau/irf_sigma, loc=irf_mean, scale=irf_sigma)
    return pdist
