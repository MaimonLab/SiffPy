"""
Methods for estimating the phase of a vector time series

All phase-alignment methods take, at the very least,
an argument vector_timeseries, which is a numpy array,
and accept a keyword argument error_estimate, which is a boolean
"""
from typing import Callable, Iterable, Union

import numpy as np
from scipy.stats import circmean

__all__ = [
    'pva'
]

def pva(vector_timeseries : np.ndarray, error_function : Union[Callable,str] = None,
    filter_fcn : Union[Callable,str] = None, **kwargs) -> np.ndarray:
    """
    Population vector average, a la Jayaraman lab

    Arguments
    ---------

    vector_timeseries : np.ndarray

        Standard for phase estimation procedures
        (call siffmath.phase_alignment_functions() for more info).

    error_function : str

        Name of error functions that can be returned. If error functions are used,
        returns a tuple, not just the pva. Can also provide a callable function for
        your own error. The function should take two arguments, the first being the
        vector_timeseries and the second being the PVA value itself. You can choose
        to ignore either argument if you want.
        
            Options:

                - relative_magnitude : estimates the inverse of the sinc function applied
                    to ( |PVA| / SUM(|COMPONENTS|) ).
                    Ranges from 0 when the population vector
                    is perfectly peaked to pi when the population 
                    vector is perfectly uniform.

    filter_fcn : function or string

        A function to apply to the time series, OR a string.
        The string specifies the function name, and if a string is used, 
        the other keyword arguments must be provided!

            Available options (and arguments):

                - 'lowpass' : NOT YET IMPLEMENTED

    """
    
    error_fcns = [
        'relative_magnitude'
    ]

    if (not error_function in error_fcns) and not (error_function is None) and not (callable(error_function)):
        raise ValueError(f"Error function request is not an available option. Available preset options are:\n{error_fcns}")

    # First normalize each row
    sorted_vals = np.sort(vector_timeseries,axis=1)
    min_val = sorted_vals[:,sorted_vals.shape[-1]//20]
    max_val = sorted_vals[:,int(sorted_vals.shape[-1]*(1.0-1.0/20))]
    vector_timeseries = ((vector_timeseries.T - min_val)/(max_val - min_val)).T

    pva_val = np.matmul(
                np.exp(np.linspace(0,2*np.pi,vector_timeseries.shape[0])*1j),
                vector_timeseries
                )

    phase = np.angle(pva_val) % (2*np.pi)

    if not (error_function is None):
        err = None
        if not callable(error_function):
            error_function = eval(error_function)
        err = error_function(vector_timeseries, pva_val)
        phase = (phase, err)

    if filter_fcn is None:
        return phase

    if callable(filter_fcn):
        return filter_fcn(phase, **kwargs)

    if type(filter_fcn) == str:

        raise ValueError(f"No filter function implemented by the name {filter_fcn}.")
    
    raise ValueError(f"filter_fcn argument {filter_fcn} is neither a callable function nor a string!")

def relative_magnitude(vector_timeseries, pva_val) -> np.ndarray:
    """
    Ranges from zero to pi, 0 when every vector component is 0 except 
    for one, and pi when every component is of equal magnitude.

    sinc(error_width) = |PVA| / sum(|PVA components|)
    """
    return arcsinc(np.abs(pva_val)/np.sum(np.abs(vector_timeseries),axis=0))

def arcsinc(z : np.ndarray) -> np.ndarray:
    """
    Inverts the Taylor series of the first four terms of the
    sinc function, i.e. the inverse of
    1 - x^2/6 + x^4/120 - x^6/5040 = z.
    
    Error is bounded above by 1.5 degrees.
    
    Solved because the cubic is invertible, and this is a
    cubic in x^2.
    """
    term_A = 2*(7**(1/3))
    term_A *= ( (5**(0.5)) * (405*(z**2)+198*z + 62)**(0.5) -
               45*z - 11
              )**(1.0/3)
    term_B = 6*(7**(2/3))
    term_B /= (
        (5**(0.5)) * (405*(z**2)+198*z+62)**(0.5) -
        45*z - 11
    ) ** (1.0/3)
    return (term_A - term_B + 14)**(0.5)