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

def chi_sq_exp(photon_arrivals : np.ndarray, param_tuple : tuple, cut_negatives : bool = True)->float:
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
    
    RETURN VALUES
    ----------
    Chi-squared : float 
        
        Chi-squared statistic of the histogram data under the model of the FLIMParams object
    """
    arrival_p = np.zeros(photon_arrivals.shape) # default, will be overwritten

    t_o = param_tuple[-2]
    tau_g = param_tuple[1]

    #iterate through components, adding up likelihood values
    n_exps = len(param_tuple-2)//2
    for exp_idx in range(n_exps):
        arrival_p += param_tuple[2*exp_idx+1] * monoexponential_prob(
            np.arange(arrival_p.shape[0])-t_o, # x_range
            param_tuple[2*exp_idx], #tau
            tau_g,
            cut_negatives=cut_negatives
        )

    total_photons = np.sum(photon_arrivals)

    chi_sq = ((photon_arrivals - total_photons*arrival_p)**2)/arrival_p

    chi_sq[np.isinf(chi_sq)]=0

    return np.nansum(chi_sq)

def monoexponential_prob(x_range : np.ndarray, tau : float, tau_g : float, cut_negatives : bool = True)->float:
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


    RETURN VALUES
    ----------

    p : np.ndarray
    
        Probability of each corresponding element of x_range
    """
    gauss_coeff = (1/(2*tau)) * np.exp((tau_g**2)/(2*(tau**2)))
    normalization = special.erfc( -(tau * x_range - tau_g**2)/ (np.sqrt(2)*tau_g*tau) )
    exp_dist = np.exp(-x_range/tau)

    if cut_negatives:
        p_out = gauss_coeff * normalization * exp_dist * (x_range > 0) # for above tau_o
    else:
        p_out = gauss_coeff * normalization * exp_dist
    
    return p_out
