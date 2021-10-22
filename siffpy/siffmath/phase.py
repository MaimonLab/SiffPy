"""
Methods for estimating the phase of a vector time series

All phase-alignment methods take, at the very least,
an argument vector_timeseries, which is a numpy array,
and accept a keyword argument error_estimate, which is a boolean
"""
from typing import Callable, Iterable, Union

import numpy as np

def pva(vector_timeseries : np.ndarray, error_estimate : bool = False, filter_fcn : Union[Callable,str] = None, **kwargs) -> np.ndarray:
    """
    Population vector average, a la Jayaraman lab

    Arguments
    ---------

    vector_timeseries : np.ndarray

        Standard for phase estimation procedures (call siffmath.phase_alignment_functions() for more info).

    error_estimate : bool

        A flag for whether or not this returns an error estimate of the phase

    filter_fcn : function or string

        A function to apply to the time series, OR a string.
        The string specifies the function name, and if a string is used, the other keyword arguments must be provided!

            Available options (and arguments):

                - 'lowpass' : 
    """
    if error_estimate:
        raise NotImplementedError("Error estimation has not yet been implemented")

    raw_phase = np.angle(
        np.matmul(
                np.exp(np.linspace(0,2*np.pi,vector_timeseries.shape[0])*1j),
                vector_timeseries)
            ) % (2*np.pi)

    if filter_fcn is None:
        return raw_phase

    if callable(filter_fcn):
        return filter_fcn(raw_phase)

    if type(filter_fcn) == str:


        raise ValueError(f"No filter function implemented by the name {filter_fcn}.")
    
    raise ValueError(f"filter_fcn argument {filter_fcn} is neither a callable function nor a string!")