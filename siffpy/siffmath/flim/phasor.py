"""
Generic functions for dealing with phasors.
"""
from typing import Any, Union
import numpy as np

from siffpy.core.flim import FLIMParams

def tau_to_phasor(tau : Union[np.ndarray,float], rep_period: float) -> Union[np.ndarray,complex]:
    """
    Returns the corresponding phasor value for a single exponential
    with the given lifetime and laser repetition period. Tau
    and the repetition period should be in the same units.

    ## Arguments

    `tau : float`
        The lifetime of the single exponential in the same units as the
        laser repetition period. May be an array of `tau` values.

    `rep_period : float`
        The laser repetition period in the same units as the lifetime.

    ## Returns

    `complex`
        The phasor value corresponding to the given lifetime and repetition period.
        Same shape as the input `tau` series.

    ## Example

    ```python
    from siffpy.siffmath.flim.phasor import tau_to_phasor

    tau = 2.5
    rep_period = 10

    phasor = tau_to_phasor(tau, rep_period)

    print(phasor)

    >>> (0.288400439142001+0.4530183504502902j)
    """
    g = (1/(1 + (2*np.pi*tau/rep_period)**2))
    s = (2*np.pi*tau/rep_period)/(1 + (2*np.pi*tau/rep_period)**2)
    return g + 1j*s

def phasor_to_tau(phasor : Union[np.ndarray,complex], rep_period : float) -> Union[np.ndarray,float]:
    """
    Converts a phasor to a tau value, assuming it corresponds to a
    single exponential with the given laser repetition period.

    ## Arguments

    `phasor : complex`
        The phasor value to convert to a lifetime. May be an array of phasor values.

    `rep_period : float`
        The laser repetition period in the same units as the returned lifetime.

    ## Returns

    `float`
        The lifetime corresponding to the given phasor and repetition period.
        Same shape as the input phasor.
    """
    g = phasor.real
    s = phasor.imag
    return rep_period*s/(g*2*np.pi)

def phasor_to_fraction(
        phasor : Union[np.ndarray[Any, np.complex128], complex],
        params : FLIMParams,
        rep_period : float,
    ) -> np.ndarray[Any, np.float64]:
    """
    Given a phasor and a set of `FLIMParams`, returns the fraction in the
    corresponding exponential states. The phasor should be corrected for
    the instrument response function and have the noise subtracted.

    ## Arguments

    `phasor : np.ndarray[Any, np.complex128]`
        The phasor values to convert to fractions. The phasor values should
        be corrected for the instrument response function and have the noise
        subtracted, otherwise when the data are projected onto the lines connecting
        states, the pull towards zero from uniformly distributed noise will distort
        the projection.

    `params : FLIMParams`
        The `FLIMParams` object to use for the conversion. The number of exponentials
        in the `FLIMParams` object determines the dimensions of the output array.

    `rep_period : float`
        The laser repetition period in the same units as the lifetimes in the `FLIMParams`
        object. Necessary to know what the phasor values correspond to.

    ## Returns

    `np.ndarray[Any, np.float64]`
        The fraction in each exponential state corresponding to the given phasor.
        Shape is size(input_phasor) x (params.n_exps - 1), with the last state in
        the `FLIMParams` object's `exps` attribute being implicit (1 - sum(fractions)).
    """

    if len(params.exps) > 2:
        raise NotImplementedError(
           "Phasor to fraction only correct for two exponentials," 
            + " at least for now!"
        )
    
    # endpoints of the line connecting the two states 
    bounds = np.array([tau_to_phasor(exp.tau, rep_period = rep_period) for exp in params.exps])
    return np.array([
        ((phasor - startpt)/(endpt - startpt)).real
        for endpt, startpt in zip(bounds[:-1], bounds[1:])
    ]).T.squeeze()
    
def universal_circle() -> np.ndarray[Any, np.complex128]:
    """
    Returns the universal circle on which single-exponential phasors live.
    A semicircle centered on (0.5, 0) of radius 0.5.

    ## Returns

    np.ndarray[Any, np.complex128]
        The x and y coordinates of the universal circle as the
        real and imaginary parts, respectively.
    """
    theta = np.linspace(0, np.pi, 500)

    return 0.5 + 0.5*np.exp(1j*theta)

def histogram_to_phasor(hist : np.ndarray[Any, np.float64]) -> np.ndarray[Any, np.complex128]:
    """
    Transforms an arrival time histogram, or a series of them,
    into a single phasor. Assumes the fastest axis of the histogram
    is the arrival time axis. You'd think the FFT would be the
    fastest way to do this, but since we just want the fundamental
    frequency and the histgram dimension is unlikely to be a power of
    2, this is faster!

    Note: this phasor is uncorrected for the instrument response function.
    
    ## Arguments

    `hist : np.ndarray[Any, np.float64]`
        The arrival time histogram or histograms to transform. The units don't
        matter -- the phasor representation is in terms of laser repetition rate.

    ## Returns

    `np.ndarray[Any, np.complex128]`
        The phasor representation of the histogram(s), with each histogram transformed
        into a single complex phasor.

    ## Example

    ```python

    import numpy as np

    from siffpy.siffmath.flim.phasor import histogram_to_phasor

    # A series of histograms with 600 bins each,
    # drawn from a mixture of two states that change
    # linearly from 0 to 1 and 1 to 0 over the course
    # of the series.
    
    hist_state_one = np.exp(-np.linspace(0,600,600)/60)/60
    hist_state_two = np.exp(-np.linspace(0,600,600)/240)/240

    frac = np.linspace(0,1,1000)

    hist = np.array([hist_state_one*(1-f) + hist_state_two*f for f in frac])

    phasor = histogram_to_phasor(hist)
    
    print(phasor)
    
    >>> [0.71969262+0.44610343j 0.71965027+0.44609588j 0.71960792+0.44608832j ...
 0.13930534+0.34254633j 0.13922574+0.34253213j 0.13914614+0.34251793j]
    
    ```


    """
    x = (hist*np.cos(np.linspace(0,2*np.pi, hist.shape[-1]))).sum(axis=-1)/hist.sum(axis=-1)
    y = (hist*np.sin(np.linspace(0,2*np.pi, hist.shape[-1]))).sum(axis=-1)/hist.sum(axis=-1)
    return x + 1j*y

def correct_phasor(
        phasor : np.ndarray[Any, np.complex128],
        params : 'FLIMParams',
        hist_length : int,
        subtract_noise : bool = False,
    ) -> np.ndarray[Any, np.complex128]:
    """
    Corrects a phasor for the effects of the instrument response function
    and projects onto the line connecting the states of the `FLIMParams` if 
    there are only two states and `project` is `True`. Casts the `FLIMParams`
    object to `countbins` units for the duration of the correction.

    ## Arguments

    `phasor : np.ndarray[Any, np.complex128]`
        The phasor representation of an arrival time histogram to correct
        using the fit `FLIMParams` object.

    `params : FLIMParams`
        The fit `FLIMParams` object to use for correction. Its offset will
        be used to rotate the phasor about the origin, and if it has
        `Exp` params those will be used to project the phasor onto a line.

        TODO: use the `noise` attribute directly! Maybe better than projecting?

    `hist_length : int`
        The length of the histogram used to compute the phasor -- i.e., the
        number of histogram bins between each laser pulse.

    `subtract_noise : bool = False`
        Whether or not to additionally project the phasor onto the line
        connecting the exponentials of the `FLIMParams` object.
        Used to subtract out the effects of noise, which push the phasor towards
        the center of the universal circle. Currently only implemented for
        two exponentials.

    ## Returns

    `np.ndarray[Any, np.complex128]`
        The corrected phasor values.
    """
    with params.as_units('countbins'):
        offset = params.tau_offset
        rotated_phasor = phasor*np.exp(-1j*2*np.pi*offset/hist_length)

        if subtract_noise:
            if len(params.exps) > 2:
                raise NotImplementedError(
                    "Projection only implemented for two exponentials! "
                    + "Problem is underdefined if more than two exponentials."
                )
            
            states = [tau_to_phasor(exp.tau, rep_period = hist_length) for exp in params.exps]
            diff = np.diff(states)
            
            #scale factor for the rotated phasor
            rotated_phasor *= (
                states[0].real * diff.imag 
                - states[0].imag * diff.real
            ) / (
                rotated_phasor.real * diff.imag 
                - rotated_phasor.imag * diff.real
            )

    return rotated_phasor