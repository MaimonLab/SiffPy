"""
Methods for estimating the phase of a vector time series

All phase-alignment methods take, at the very least,
an argument vector_timeseries, which is a numpy array,
and accept a keyword argument error_estimate, which is a boolean
"""
from typing import Callable, Union, Optional, Tuple, Any
from enum import Enum

import numpy as np
from scipy.special import i0, i1

from siffpy.core.utils.types import ComplexArray, FloatArray
from siffpy.siffmath.phase.traces import (
    PhaseTrace,
)
from siffpy.siffmath.fluorescence import FluorescenceTrace
from siffpy.siffmath.flim import FlimTrace
from siffpy.siffmath.phase.phase_analyses import (
    estimate_phase, fit_offset, sliding_correlation, multiscale_circ_corr # noqa: F401
)

__all__ = [
    'pva',
    'pva_flim',
]

class PhaseErrorFunction(Enum):
    """
    Enumerates the error functions available for phase estimation.
    """
    relative_magnitude = 'relative_magnitude'

def pva_flim(
        vector_timeseries : Union[np.ndarray, FlimTrace],
        lifetime_bounds : Tuple[float, float],
        time : Optional[np.ndarray] = None,
        error_function : Optional[Union[Callable,str]] = 'relative_magnitude',
        filter_fcn : Optional[Callable] = None,
        **kwargs
) -> PhaseTrace:
    """
    Computes the PVA of a FLIM trace, using the lifetime bounds to
    determine the phase of timepoint.

    Arguments
    ---------

    vector_timeseries : np.ndarray (even better if a FlimTrace)
    
            Standard for phase estimation procedures
            (call siffmath.phase_alignment_functions() for more info).

    lifetime_bounds : Tuple[float, float]

        The bounds of the sensor lifetime (in whatever units are passed in).
        (Lower, upper)

    time : np.ndarray

        A time axis. May be None to just be ignored.

    error_function : str

        Name of error functions that can be returned. If error functions are used,
        returns a tuple, not just the pva. Can also provide a callable function for
        your own error. The function should take two arguments, the first being the
        vector_timeseries and the second being the PVA value itself. You can choose
        to ignore either argument if you want.

    filter_fcn : function

        A function to apply to the time series before computing the PVA.

    Returns
    -------

    pva_val : PhaseTrace

        The phase trace of the PVA. If error_function is not None,
        the PhaseTrace will have an attribute `error_array` that
        contains the lower and upper bounds of the error for
        each timepoint.
    
    """
    if isinstance(vector_timeseries, FlimTrace):
        vector_timeseries = vector_timeseries.lifetime
    vector_timeseries = (vector_timeseries - lifetime_bounds[0])/(lifetime_bounds[1] - lifetime_bounds[0])
    #TODO: Do something smarter with the intensity stuff if it's a FLIMTrace
    return pva(
        vector_timeseries,
        time = time,
        error_function = error_function,
        filter_fcn = filter_fcn,
        **kwargs
    )

def pva(
        vector_timeseries : Union[np.ndarray, FluorescenceTrace],
        normalize        : bool                              = True, 
        time              : Optional[np.ndarray]              = None,
        error_function    : Optional[Union[Callable,str]]     = 'relative_magnitude',
        filter_fcn        : Optional[Union[Callable,str]]     = None,
        angle_coords      : Optional[np.ndarray[Any, Any]]              = None,
        **kwargs
    ) -> PhaseTrace:
    """
    Population vector average, a la Jayaraman lab

    Arguments
    ---------

    vector_timeseries : np.ndarray (even better if a FluorescenceVector)

        Standard for phase estimation procedures
        (call siffmath.phase_alignment_functions() for more info).

    normalize : bool
        Whether to normalize the vector timeseries before computing the PVA.

    time : np.ndarray

        A time axis. May be None to just be ignored.

    error_function : str

        Name of error functions that can be returned. Can also provide a callable function for
        your own error. The function should take two arguments, the first being the
        vector_timeseries and the second being the PVA value itself. You can choose
        to ignore either argument if you want.
        
            Options:

                - relative_magnitude : estimates the inverse of the sinc function applied
                    to ( |PVA| / SUM(|COMPONENTS|) ).
                    Ranges from 0 when the population vector
                    is perfectly peaked to pi when the population 
                    vector is perfectly uniform.

    filter_fcn : function

        A function to apply to the time series before computing the PVA.

    angle_coords : np.ndarray

        The angular coordinates of the first dimension of the vector timeseries,
        i.e. the angle that each vector component corresponds to. If not provided,
        will assume that the angle is linearly spaced between -pi and pi. Can
        provide complex numbers on the unit circle or floats corresponding to an
        angle in radians.
    
    """
    
    if normalize:
        sorted_vals = np.sort(vector_timeseries,axis=1)
        min_val = sorted_vals[:,sorted_vals.shape[-1]//20]
        max_val = sorted_vals[:,int(sorted_vals.shape[-1]*(1.0-1.0/20))]
        vector_timeseries = ((vector_timeseries.T - min_val)/(max_val - min_val)).T

    if isinstance(vector_timeseries, FluorescenceTrace) and (angle_coords is None):
        if (
            isinstance(vector_timeseries.angle, np.ndarray) 
            and (vector_timeseries.angle.dtype == np.complex128)
        ):
            angle_coords = vector_timeseries.angle
        elif (
            (vector_timeseries.angle is not None)
            and all(x is not None for x in vector_timeseries.angle)
        ):
            angle_coords = np.exp(-1j*vector_timeseries.angle)
    elif angle_coords is None:
        angle_coords = np.exp(np.linspace(np.pi, -np.pi, vector_timeseries.shape[0])*1j) # it goes clockwise.
    
    if angle_coords.dtype != np.complex128:
        angle_coords = np.exp(1j*angle_coords)

    pva_val = np.asarray(
        np.matmul(
            angle_coords,
            vector_timeseries
        )
    )

    if isinstance(filter_fcn, str):
        raise ValueError(f"No filter function implemented by the name {filter_fcn}.")

    if callable(filter_fcn):
        pva_val = filter_fcn(pva_val, **kwargs)
        try:
            vector_timeseries = filter_fcn(vector_timeseries, **kwargs)
        except Exception: # Take care of yourself this time.
            pass

    if not callable(error_function):
        try:
            error_function = PhaseErrorFunction(error_function)
        except ValueError:
            raise ValueError(
                f"Error function {error_function} not recognized." +
                f" Available options are {PhaseErrorFunction.__members__}"
            )
        if error_function == PhaseErrorFunction.relative_magnitude:
            error_function = relative_magnitude
    
    err = error_function(vector_timeseries, pva_val)

    return PhaseTrace(
        pva_val,
        method = 'pva_normalized' if normalize else 'pva',
        error_array = err,
        time = time
    )

def load_interp_func()->Callable:
    """ Loads the stored interp function """
    import importlib_resources
    lookup_path = str(
        importlib_resources.files('siffpy.siffmath.phase')
        .joinpath('relative_mag_lookup.npz')
    )
    with np.load(lookup_path) as loaded:
        theta = loaded['theta']
        magnitude = loaded['magnitude']
    from scipy.interpolate import interp1d
    return interp1d(magnitude, theta)

INTERP_FUNC = load_interp_func()

def relative_magnitude(
        vector_timeseries : FloatArray,
        pva_val : ComplexArray
    ) -> np.ndarray:
    """
    Ranges from zero to pi, 0 when every vector component is 0 except 
    for one, and pi when every component is of equal magnitude.

    The error function is stored as a lookup table, because I don't have
    an analytic expression for it. It's an interpolation between an 
    arcsinc function (the exact expression for a rect signal) and a numerical
    inverse of the relative magnitude of a von Mises distribution (which
    can be computed from I1(kappa)/I0(kappa), and translating kappa to a
    full-width half maximum).

    For details, look at the source (and/or README).

    """
    z = np.abs(pva_val)/np.sum(np.abs(vector_timeseries),axis=0)
    return INTERP_FUNC(z)

### BELOW IS JUST ILLUSTRATIVE, NOT USED IN THE CODE ###
def fwhm_von_mises(kappa : float)->float:
    """
    Returns the full width at half maximum of a von Mises distribution
    with concentration parameter kappa.
    """
    return np.arccos(1-np.log(2)/kappa)

def relative_mag_vm(kappa : float) -> float:
    """
    Returns the relative magnitude error function for a von Mises distribution
    with concentration parameter kappa.
    """
    return i1(kappa)/i0(kappa)

def arcsinc(z : np.ndarray) -> np.ndarray:
    """
    Inverts the Taylor series of the first four terms of the
    sinc function, i.e. the inverse of
    1 - x^2/6 + x^4/120 - x^6/5040 = z.
    
    Error is bounded above by 1.5 degrees.
    
    Solved because the cubic is invertible, and this is a
    cubic in x^2.

    Provides the FWHM of a rect signal where the relative magnitude calculation
    produces a value of "z".
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