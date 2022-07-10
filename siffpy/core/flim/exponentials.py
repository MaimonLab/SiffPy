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
from scipy import special

def chi_sq_exp(photon_arrivals : np.ndarray, param_tuple : tuple, negative_scope : float = 0.0)->float:
    """
    Returns the chi-squared statistic
    of a photon arrival data set, given
    parameters in params

    INPUTS
    ----------
    photon_arrival : np.ndarray

        Histogrammed arrival time of each photon. Element n is the count of photons arriving in histogram bin n.

    param_tuple : tuple
        
        A tuple consistent with the FLIMParams property "param_tuple"

    negative_scope : float

        As in monoexponential_prob.
    
    RETURN VALUES
    ----------
    Chi-squared : float 
        
        Chi-squared statistic of the histogram data under the model of the FLIMParams object
    """
    arrival_p = np.zeros_like(photon_arrivals, dtype=float) # default, will be overwritten

    t_o = param_tuple[-2]
    tau_g = param_tuple[1]

    #iterate through components, adding up likelihood values
    n_exps = (len(param_tuple)-2)//2
    for exp_idx in range(n_exps):
        arrival_p += param_tuple[2*exp_idx+1] * monoexponential_prob(
            np.arange(arrival_p.shape[0], dtype=float)-t_o, # x_range
            param_tuple[2*exp_idx], #tau
            tau_g,
            negative_scope = negative_scope
        )

    # Don't include the points outside of the negative_scope parameter
    bottom_bin = np.floor(negative_scope/tau_g)
    #bottom_bin = int(max(t_o - bottom_bin, 0))
    #bottom_bin = int(max(t_o, 0))
    #photon_arrivals = photon_arrivals[bottom_bin:]
    #arrival_p = arrival_p[bottom_bin:]

    total_photons = np.sum(photon_arrivals)

    chi_sq = ((photon_arrivals - total_photons*arrival_p)**2)/(total_photons*arrival_p)

    chi_sq[np.isinf(chi_sq)]=0

    return np.nansum(chi_sq)

def monoexponential_prob(x_range : np.ndarray, tau : float, tau_g : float, negative_scope : float = 0.0)->float:
    """
    
    Takes in parameters of an exponential distribution
    convolved with a Gaussian, and outputs the probability
    of each element of x_range.

    Presumes the temporal offset has been subtracted out already!
    TODO: Switch from pointwise estimate to bin-size integral
    
    
    INPUTS
    ----------

    x_range : np.ndarray
        
        The range of values to compute the probability of.

    tau : float
    
        The decay parameter of the exponential (IN UNITS OF BINS OF THE ARRAY)

    tau_g : float
    
        Width of the Gaussian with which the exponential
        distribution has been convolved (IN UNITS OF BINS OF THE ARRAY)

    negative_scope : float

        How many units of tau_g to go in the negative direction in provided the
        probability distribution. E.g. negative_scope = 1.0 means will also provide
        a non-zero number for up to tau_g bins before the 0 point. If negative_scope < 0,
        then 0 is used.

    RETURN VALUES
    ----------

    p : np.ndarray
    
        Probability of each corresponding element of x_range
    """
    gauss_coeff = (1/(2*tau)) * np.exp((tau_g**2)/(2*(tau**2)))
    normalization = special.erfc( -(tau * x_range - tau_g**2)/ (np.sqrt(2)*tau_g*tau) )
    exp_dist = np.exp(-x_range/tau)

    if negative_scope > 0.0:
        p_out = gauss_coeff * normalization * exp_dist * (x_range > -(negative_scope/tau_g)) # for above tau_o
    else:
        p_out = gauss_coeff * normalization * exp_dist * (x_range > 0)
    
    return p_out
