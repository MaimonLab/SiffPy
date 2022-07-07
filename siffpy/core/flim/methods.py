"""
Methods for analysis of FLIM data

- SCT June 2022
"""
import logging

import numpy as np

from .fluorophore_inits import available_fluorophores
from .flimparams import FLIMParameter, FLIMParams, Exp, Irf
from .flimunits import FlimUnits

def channel_exp_fit(
        photon_arrivals : np.ndarray,
        num_components : int = 2,
        initial_fit : tuple[FLIMParameter] = None,
        color_channel : int = None,
        **kwargs
    ) -> FLIMParams:
    """
    Takes row data of arrival times and returns the param dict from an exponential fit.
    TODO: Provide more options to how fitting is done


    INPUTS
    ----------

    photon_arrivals (1-dim ndarray): Histogrammed arrival time of each photon.

    num_components (int): Number of components to the exponential TODO: enable more diversity?

    initial_fit (tuple): FLIMParameter iterable


    RETURN VALUES
    ----------
    FLIMParams -- (FLIMParams object)
    """
    if (initial_fit is None):
        if num_components == 2:
            initial_fit = (
                Exp(tau = 115, frac = 0.7),
                Exp(tau = 25, frac = 0.3),
                Irf(tau_offset = 100.0, tau_g = 4.0)
            ) # pretty decent guess for Camui data

        if num_components == 1:
            initial_fit = (
                Exp(tau = 140, frac = 1.0),
                Irf(tau_offset = 100.0, tau_g = 4.0)
            ) # GCaMP / free GFP fluoroscence

    noise = 0.0
    if 'use_noise' in kwargs:
        noise = 0.01*kwargs['use_noise'] # should do this right... TODO

    params = FLIMParams(
        *initial_fit,
        color_channel = color_channel,
        units = FlimUnits.COUNTBINS,
        noise = noise
    )

    params.fit_to_data(
        photon_arrivals,
        num_exps=num_components,
        initial_guess=params.param_tuple
    )
    
    if params.chi_sq(photon_arrivals) == 0:
        logging.warn("Returned FLIMParams object has a chi-squared of zero, check the fit to be sure!")
    
    return params

def fit_exp(
        histograms : np.ndarray,
        num_components: 'int|list[int]' = 2,
        fluorophores : list[str] = None,
        use_noise : bool = False
    ) -> list[FLIMParams]:

    """
    params = siffpy.fit_exp(histograms, num_components=2)


    Takes a numpy array with dimensions n_colors x num_arrival_time_bins
    returns a color-element list of dicts with fits of the fluorescence emission model for each color channel

    INPUTS
    ------
    histograms: (ndarray) An array of data with the following shapes:
        (num_bins,)
        (1, num_bins)
        (n_colors, num_bins)

    num_components: 
    
        (int) Number of exponentials in the fit (one color channel)
        (list) Number of exponentials in each fit (per color channel)

        If there are more color channels than provided in this list (or it's an
        int with a multidimensional histograms input), they will be filled with
        the value 2.

    fluorophores (list[str] or str):

        List of fluorophores, in same order as color channels. By default, is None.
        Used for initial conditions in fitting the exponentials. I doubt it's critical.

    use_noise (bool, optional):

        Whether or not to put noise in the FLIMParameter fit by default
    
    RETURN VALUES
    -------------

    fits (list):
        A list of FLIMParams objects, containing fit parameters as attributes and with functionality
        to return the parameters as a tuple or dict.

    """

    # n_colors = 1
    # if len(histograms.shape)>1:
    #     n_colors = histograms.shape[0]

    # if n_colors > 1:
    #     if not (type(num_components) == list):
    #         num_components = [num_components]
    #     if len(num_components) < n_colors:
    #         num_components += [2]*(n_colors - len(num_components)) # fill with 2s

    # # type-checking -- HEY I thought this was Python!
    # if not (isinstance(fluorophores, list) or isinstance(fluorophores, str)):
    #     fluorophores = None

    # # take care of the fluorophores arg
    # if fluorophores is None:
    #     fluorophores = [None] * n_colors

    # if len(fluorophores) < n_colors:
    #     fluorophores += [None] * (n_colors - len(fluorophores)) # pad with Nones

    # # take these strings, turn them into initial guesses for the fit parameters
    # availables = available_fluorophores(dtype=dict)

    # for idx in range(len(fluorophores)):
    #     if not (fluorophores[idx] in availables):
    #         logging.warning("\n\nProposed fluorophore %s not in known fluorophore list. Using default params instead\n" % (fluorophores[idx]))
    #         fluorophores[idx] = None

    # list_of_dicts_of_fluorophores = [availables[tool_name] if isinstance(tool_name,str) else None for tool_name in fluorophores]
    # list_of_flimparams = [FLIMParams(param_dict = this_dict, use_noise = use_noise) if isinstance(this_dict, dict) else None for this_dict in list_of_dicts_of_fluorophores]
    # fluorophores_dict_list = [FlimP.param_dict() if isinstance(FlimP, FLIMParams) else None for FlimP in list_of_flimparams]

    n_colors = 1
    if len(histograms.shape) > 1:
        n_colors = len(histograms)

    if n_colors > 1:
        fit_list = [channel_exp_fit( histograms[x], num_components = num_components[x], initial_fit = None , color_channel = x, use_noise = use_noise) for x in range(n_colors)]
    else:
        fit_list = [channel_exp_fit( histograms, num_components = num_components, initial_fit = None, use_noise = use_noise )]

    return fit_list
