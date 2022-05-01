"""
Analyses on extracted phase information
"""

from typing import Union
import random

import numpy as np
from scipy.stats import circmean

from ...siffutils.circle_fcns import *
from .traces import PhaseTrace

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

def sliding_correlation(
        trace_1 : np.ndarray,
        trace_2 : np.ndarray,
        window_width : float,
    )->np.ndarray:
    """
    Returns a timeseries of the sliding circular correlation between two circular
    signals.
    """
    trace_1 = np.squeeze(trace_1)
    trace_2 = np.squeeze(trace_2)
    if (trace_1.ndim != 1) or (trace_2.ndim != 1):
        raise ValueError("Arguments must have only one non-singleton dimension")

    if not (trace_1.shape == trace_2.shape):
        raise ValueError("Arguments must have same shape")

    from numpy.lib.stride_tricks import sliding_window_view

    return circ_corr(
                sliding_window_view(trace_1, window_width, axis = 0),
                sliding_window_view(trace_2, window_width, axis = 0),
                axis=1,
            ).flatten()