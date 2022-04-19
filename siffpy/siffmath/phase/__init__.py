"""
Methods for estimating the phase of a vector time series

All phase-alignment methods take, at the very least,
an argument vector_timeseries, which is a numpy array,
and accept a keyword argument error_estimate, which is a boolean
"""
from typing import Callable, Iterable, Union
import random

import numpy as np
from scipy.stats import circmean
from ...siffutils.circle_fcns import circ_interpolate_between_endpoints

from .traces import *
from ..fluorescence import FluorescenceTrace
#from ..fluorescence import FluorescenceVector

__all__ = [
    'fit_offset',
    'pva'
]

def fit_offset(
        fictrac_trace : np.ndarray,
        phase_trace : Union[PhaseTrace,np.ndarray],
        fictrac_time : np.ndarray = None,
        phase_time : np.ndarray = None,
        t_bounds : tuple = None,
        N_points : int = 1000,
    )->float:
    """
    Computes a SINGLE float-valued offset between a FicTrac trace and a PhaseTrace
    (or a numpy array of phases). Accepts time axes for each, though if the PhaseTrace
    has a time attribute, it will prefer that.

    If argument t_bounds is provided, will bound the time period over which the offset
    is estimated to the region within the bounds provided. Otherwise uses the full range
    of the smallest shared region between the phase time and fictrac time.

    Returns FICTRAC - PHASE angle
    """

    # Arg parsing for a little while..

    if isinstance(phase_trace, PhaseTrace):
        if not phase_trace.time is None:
            phase_time = phase_trace.time
        phase_trace = np.asarray(phase_trace)
    
    if (fictrac_time is None) or (phase_time is None):
        raise ValueError(
            "Time values for both the FicTrac data and phase data must be present." +
            " If this information is not present in the PhaseTrace, pass a numpy array to "+
            "the optional parameter `phase_time"
        )

    if t_bounds is None:
        t_bounds = list((
            max(np.min(phase_time), np.min(fictrac_time)),
            min(np.max(phase_time), np.max(fictrac_time))
        ))
    
    if not hasattr(t_bounds, '__iter__'):
        raise ValueError("t_bounds must be an iterable or None")
    if len(t_bounds) > 2:
        raise ValueError("t_bounds can only contain a minumum and/or maximum element.")

    t_bounds = list(t_bounds)

    if t_bounds[0] is None:
        t_bounds[0] = max(np.min(phase_time), np.min(fictrac_time))
    if t_bounds[1] is None:
        t_bounds[1] = min(np.max(phase_time), np.max(fictrac_time))

    # Okay now I have a nice clean set of arguments to interpolate with.

    # random timepoints
    timepoints = np.array([random.uniform(*t_bounds) for x in range(N_points)])

    # get the nearest phase points to interpolate between
    phase_idxs = [np.argpartition(np.abs(timepoints[x]-phase_time),2)[:2] for x in range(len(timepoints))]
    phasetime_endpts = phase_time[phase_idxs]
    phase_vals = phase_trace[phase_idxs]

    # get the nearest fictrac points to interpolate between
    fic_idxs = [np.argpartition(np.abs(timepoints[x]-fictrac_time),2)[:2] for x in range(len(timepoints))]
    fictractime_endpoints = fictrac_time[fic_idxs]
    fictrac_vals = fictrac_trace[fic_idxs]

    # now interpolate the phase
    phase_interp = circ_interpolate_between_endpoints(
        timepoints,
        phasetime_endpts,
        phase_vals
    )

    #interpolate the fictrac values
    fictrac_interp = circ_interpolate_between_endpoints(
        timepoints,
        fictractime_endpoints,
        fictrac_vals
    )

    return circmean(np.angle(np.exp(fictrac_interp*1j)/np.exp(phase_interp*1j)))

def pva(
        vector_timeseries : Union[np.ndarray, FluorescenceTrace], time : np.ndarray = None,
        error_function : Union[Callable,str] = None,
        filter_fcn : Union[Callable,str] = None, **kwargs
    ) -> PhaseTrace:
    """
    Population vector average, a la Jayaraman lab

    Arguments
    ---------

    vector_timeseries : np.ndarray (even better if a FluorescenceVector)

        Standard for phase estimation procedures
        (call siffmath.phase_alignment_functions() for more info).

    time : np.ndarray

        A time axis. May be None to just be ignored.

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

    angle_coords = np.exp(np.linspace(2*np.pi, 0 , vector_timeseries.shape[0])*1j) # it goes clockwise.
    if isinstance(vector_timeseries, FluorescenceTrace):
        if all(x is not None for x in vector_timeseries.angle):
            angle_coords = np.exp(vector_timeseries.angle[::-1]*1j)
    
    pva_val = np.asarray(
                np.matmul(
                    angle_coords,
                    vector_timeseries
                )
            )

    phase = np.angle(pva_val) % (2*np.pi)

    if not (error_function is None):
        raise NotImplementedError("Haven't implemented error functions properly yet")
        err = None
        if not callable(error_function):
            error_function = eval(error_function)
        err = error_function(vector_timeseries, pva_val)
        phase = (phase, err)

    if callable(filter_fcn):
        phase = filter_fcn(phase, **kwargs)

    if type(filter_fcn) == str:

        raise ValueError(f"No filter function implemented by the name {filter_fcn}.")

    return PhaseTrace(phase, method = 'pva', time = time)

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