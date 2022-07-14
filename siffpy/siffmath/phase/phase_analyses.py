"""
Analyses on extracted phase information
"""

from typing import Union, Iterable
import random

import numpy as np
from scipy.stats import circmean
from numpy.lib.stride_tricks import sliding_window_view

from ...core.utils import circle_fcns
from .traces import PhaseTrace
from . import phase_estimates

def estimate_phase(vector_series : np.ndarray, *args, method='pva', error_estimate = False, **kwargs)->np.ndarray:
    """
    Takes a time series of vectors (dimensions x time) and, assuming they
    correspond to a circularized signal, returns a 'phase' estimate, according
    to the method selected.

    Arguments
    ---------

    vector_series : np.ndarray

        An array of shape (D,T) where T is time bins and D is discretized elements of a 1-dimensional ring.

    method : str

        Available methods:

            - pva : Population vector average

    """

    if error_estimate:
        raise NotImplementedError("Error estimate on the phase is not yet implemented.")

    try:
        if not callable(getattr(phase_estimates, method)): # check that the method IS callable
            raise ValueError(f"No phase estimate method {method} in SiffMath module {phase_estimates}")
    except AttributeError as e:
        raise ValueError(f"No phase estimate method {method} in SiffMath module {phase_estimates}." +
        "To see available methods, call siffmath.phase_alignment_functions()")

    phase_method = getattr(phase_estimates, method)
    return phase_method(vector_series, *args, error_estimate = error_estimate, **kwargs)

def fit_offset(
        fictrac_trace   : np.ndarray,
        phase_trace     : Union[PhaseTrace,np.ndarray],
        fictrac_time    : np.ndarray    = None,
        phase_time      : np.ndarray    = None,
        t_bounds        : tuple         = None,
        N_points        : int           = 1000,
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
    # phase_idxs = [np.argpartition(np.abs(timepoints[x]-phase_time),2)[:2] for x in range(len(timepoints))]
    # phasetime_endpts = phase_time[phase_idxs]
    # phase_vals = phase_trace[phase_idxs]

    # # get the nearest fictrac points to interpolate between
    # fic_idxs = [np.argpartition(np.abs(timepoints[x]-fictrac_time),2)[:2] for x in range(len(timepoints))]
    # fictractime_endpoints = fictrac_time[fic_idxs]
    # fictrac_vals = fictrac_trace[fic_idxs]

    # # now interpolate the phase
    # phase_interp = circle_fcns.circ_interpolate_between_endpoints(
    #     timepoints,
    #     phasetime_endpts,
    #     phase_vals
    # )

    # #interpolate the fictrac values
    # fictrac_interp = circle_fcns.circ_interpolate_between_endpoints(
    #     timepoints,
    #     fictractime_endpoints,
    #     fictrac_vals
    # )

    phase_interp = circle_fcns.circ_interpolate_to_sample_points(timepoints, phase_time, phase_trace)
    fictrac_interp = circle_fcns.circ_interpolate_to_sample_points(timepoints, fictrac_time, fictrac_trace)

    return circmean(np.angle(np.exp(fictrac_interp*1j)/np.exp(phase_interp*1j)))

def align_two_circ_vars_timepoints(
        data_1         : np.ndarray,
        data_1_time    : np.ndarray,
        data_2         : np.ndarray,
        data_2_time    : np.ndarray
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsamples to the shortest trace, circ-linearly interpolates
    the longer trace, returns the two and their shared timebase.

    Returns
    -------

    (shared_time, aligned_phase, aligned_heading)
    """

    # Figure out which to downsample

    data_list = [data_1, data_2]
    time_list = [data_1_time, data_2_time]

    downsample_idx = len(data_1_time) <= len(data_2_time) # 1 if data_2 is to be downsampled

    data_list[downsample_idx] = circle_fcns.circ_interpolate_to_sample_points(
        time_list[not downsample_idx],
        time_list[downsample_idx],
        data_list[downsample_idx]
    )

    return (time_list[not downsample_idx], *data_list)

def sliding_correlation(
        trace_1         : np.ndarray,
        trace_2         : np.ndarray,
        window_width    : int,
    )->np.ndarray:
    """
    Returns a timeseries of the sliding circular correlation between two circular
    signals. They must be sampled at the same rate! If they're not aligned, use
    phase_analyses.align_two_circ_vars_timepoints(trace1,trace1time,trace2,trace2time)
    """
    trace_1 = np.squeeze(trace_1)
    trace_2 = np.squeeze(trace_2)
    if (trace_1.ndim != 1) or (trace_2.ndim != 1):
        raise ValueError("Arguments must have only one non-singleton dimension")

    if not (trace_1.shape == trace_2.shape):
        raise ValueError("Arguments must have same shape")

    return circle_fcns.circ_corr(
                sliding_window_view(trace_1, window_width, axis = 0),
                sliding_window_view(trace_2, window_width, axis = 0),
                axis=1,
            ).flatten()

def multiscale_circ_corr(
        trace_1         : np.ndarray,
        trace_2         : np.ndarray,
        window_widths   : Iterable[int],
    )->list[np.ndarray]:
    """
    Returns a sliding window computation of the circular correlation
    between two traces across a selection of window widths. Calls
    siffpy.siffmath.phase.phase_analyses.sliding_correlation

    Arguments
    --------

    trace_1 : np.ndarray

        The first of two circular variables to correlation

    trace_2 : np.ndarray

        The second of two circular variables to correlate.

    windows : Iterable[float]

        Iterable of all windows to apply. Windows must be in units
        of ELEMENTS OF THE TRACES, not units of time.

    Returns
    -------

    circ_corrs : list[np.ndarray]

        A list of numpy arrays, one for each element of 'windows'.
        Each numpy array is of length t - w_n, where w_n is the
        value of the nth element of 'windows'.
    """
    return [sliding_correlation(
            trace_1,
            trace_2,
            w
        )
        for w in window_widths
    ]