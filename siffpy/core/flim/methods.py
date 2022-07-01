"""
Methods for analysis of FLIM data

- SCT June 2022
"""
import numpy as np

from .flimparams import FLIMParams

def fit_flim_params(histogram : np.ndarray, initial_guess = None, **kwargs)->FLIMParams:
    """
    Takes a histogram of arrival times and returns a best guess of the photon
    arrival time distribution, under the constraints provided.

    Always check the fit against the histogram data itself to be sure it looks good
    and makes sense!!

    TODO: Document and implement!
    """
    raise NotImplementedError()
