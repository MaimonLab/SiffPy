import numpy as np
from typing import Iterable

from siffpy.core.utils.circle_fcns import circ_corr_complex, running_circ_corr_complex
from .traces import PhaseTrace

def circular_correlate(x : PhaseTrace, y : PhaseTrace, method : str = 'Fisher', ) -> np.ndarray:
    """
    Circular correlation of two phase traces -- does NOT take error functions into
    account. TODO: use the magnitude rather than just the angle for a more robust
    estimate of error.

    ## Arguments

    x : PhaseTrace
        The first phase trace to correlate.

    y : PhaseTrace
        The second phase trace to correlate.

    method : str

        The method to use for the correlation. Options are:
        - 'Fisher'
        - 'Jammalamadaka

    ## Returns

    np.ndarray
        The circular correlation of the two phase traces. A single number.
    """

    return circ_corr_complex(
        np.exp(1j*x.angle),
        np.exp(1j*y.angle),
        method = method
    )

def correlate(x : PhaseTrace, y : PhaseTrace, dts : Iterable, method : str = 'Fisher', mode : str = 'valid') -> np.ndarray:
    """
    Compute the circular cross-correlation of two phase traces, the correlation of `x` with a time-shifted
    version of `y`. Does NOT take error functions or the magnitude of the phase into account.

    ## Arguments

    x : PhaseTrace
        The first phase trace to correlate.

    y : PhaseTrace
        The second phase trace to correlate.

    method : str

        The method to use for the correlation. Options are:
        - 'Fisher'
        - 'Jammalamadaka

    mode : str

        The mode to use for the correlation. Options are:
        - 'full'
        - 'valid'
        - 'same'

    ## Returns

    np.ndarray
        The cross-circular-correlation of the two phase traces.
    """
    if not mode == 'valid':
        raise NotImplementedError("Only 'valid' mode is currently supported")
    
    x_angle = np.exp(1j*x.angle)
    y_angle = np.exp(1j*y.angle)

    raise NotImplementedError("This function is not yet implemented")