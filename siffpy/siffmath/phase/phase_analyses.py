"""
Analyses on extracted phase information
"""

from typing import Union, Iterable, List, TYPE_CHECKING
import random

import numpy as np
from scipy.stats import circmean
from numpy.lib.stride_tricks import sliding_window_view

from siffpy.core.utils import circle_fcns
from siffpy.siffmath.phase.traces import PhaseTrace
import siffpy.siffmath.phase.phase_estimates as phase_estimates

if TYPE_CHECKING:
    from siffpy.siffmath.utils.types import PhaseTraceLike

def estimate_phase(
    vector_series : np.ndarray,
    *args,
    method='pva',
    error_estimate : bool = False,
    **kwargs
    )->np.ndarray:
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

    phase_method = getattr(phase_estimates, method)
    return phase_method(vector_series, *args, error_estimate = error_estimate, **kwargs)

def fit_offset(
        fictrac_trace   : 'PhaseTraceLike',
        phase_trace     : 'PhaseTraceLike',
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

    phase_interp = circle_fcns.circ_interpolate_to_sample_points(timepoints, phase_time, phase_trace)
    fictrac_interp = circle_fcns.circ_interpolate_to_sample_points(timepoints, fictrac_time, fictrac_trace)

    return circmean(np.angle(np.exp(fictrac_interp*1j)/np.exp(phase_interp*1j)))

def sliding_correlation(
        trace_1         : 'PhaseTraceLike',
        trace_2         : 'PhaseTraceLike',
        window_width    : int,
    )->np.ndarray:
    """
    Returns a timeseries of the sliding circular correlation between two circular
    signals. They must be sampled at the same rate! If they're not aligned, use
    phase_analyses.align_two_circ_vars_timepoints(trace1,trace1time,trace2,trace2time)
    """
    trace_1 = trace_1.angle if isinstance(trace_1, PhaseTrace) else trace_1
    trace_2 = trace_2.angle if isinstance(trace_2, PhaseTrace) else trace_2

    trace_1 = np.squeeze(trace_1)
    trace_2 = np.squeeze(trace_2)
    if (trace_1.ndim != 1) or (trace_2.ndim != 1):
        raise ValueError("Arguments must have only one non-singleton dimension")

    if not (trace_1.shape == trace_2.shape):
        raise ValueError("Arguments must have same shape")

    expd_1 = np.exp(1j*trace_1)
    expd_2 = np.exp(1j*trace_2)

    return circle_fcns.running_circ_corr_complex(
        expd_1,
        expd_2,
        window_width,
        axis = 0
    )

def multiscale_circ_corr(
        trace_1         : 'PhaseTraceLike',
        trace_2         : 'PhaseTraceLike',
        window_widths   : Iterable[int],
    )->List[np.ndarray]:
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

    circ_corrs : List[np.ndarray]

        A list of numpy arrays, one for each element of 'windows'.
        Each numpy array is of length t - w_n, where w_n is the
        value of the nth element of 'windows'.
    """

    return [
        sliding_correlation(
            trace_1,
            trace_2,
            w
        )
        for w in window_widths
    ]