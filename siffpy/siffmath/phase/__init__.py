import inspect
import textwrap

import numpy as np

from siffpy.siffmath.phase.traces import PhaseTrace
import siffpy.siffmath.phase.phase_estimates as phase_estimates
from siffpy.siffmath.utils.types import PhaseTraceLike

def phase_alignment_functions(print_docstrings : bool = True)->None:
    """
    Prints the available methods for aligning a vector time series to a phase,
    as well as returning the string
    """
    print_string = ""
    memberfcns = inspect.getmembers(phase_estimates, inspect.isfunction)

    for member_fcn_info in memberfcns:
        fcn_name = member_fcn_info[0]
        fcn_call = member_fcn_info[1]
        print_string += f"\033[1m{fcn_name}\033[0m\n\n"
        print_string += f"\t{fcn_name}{inspect.signature(fcn_call)}\n\n"
        print_string += textwrap.indent(str(inspect.getdoc(fcn_call)),"\t\t")
        print_string += "\n\n"
    
    if print_docstrings:
        print(print_string)
    else:
        return memberfcns
    
def phase_shift(x : np.ndarray, shift : PhaseTraceLike)->np.ndarray:
    """
    Shifts the phase of a vector time series by the phase requested.

    Expects the first dimension of x to be the phase shifting dimension.

    Interpolates for phases that do not evenly divide the number of bins,
    (e.g. if there are 4 bins and the phase is 60 degrees, 2/3 of the mass
    of the 0th bin will go into the 90 degree bin, and 1/3 will go into the 0
    degree bin).

    *TO DO: THIS IS SLOW AND BADLY IMPLEMENTED -- DO IT RIGHT, WITHOUT A LOOP*

    Parameters
    ----------

    x : np.ndarray
        The vector time series to shift the phase of. Must be of shape
        (n_bins, n_time)

    phase : PhaseTraceLike
        Accepts either an array of angles (does not need to be wrapped)
        or a `PhaseTrace` object.

    Returns
    -------
    np.ndarray
        The phase shifted vector time series. Will be of same shape as `x`

    Example
    -------

    ```python

    import numpy as np
    from siffpy.siffmath.phase import phase_shift

    x = np.random.randn(16, 12000)
    d_phase = np.random.randn(12000) + 0.01
    phase = np.angle(np.exp(1j*np.cumsum(d_phase)))
    shifted = phase_shift(x, phase)

    ```
    """
    if len(shift) != x.size//x.shape[0]:
        raise ValueError(
            "`shift` must have the same number of elements as the first dimension of `x` \
            ``` \
            assert len(shift) == x.size//x.shape[0]\
            ```"
        )
    if isinstance(shift, PhaseTrace):
        shift = np.angle(shift)

    shifted = np.zeros_like(x)
    n_cols = x.shape[0]
    for t in range(x.shape[1]):
        idx = np.angle(np.exp(1j*shift[t]))*n_cols/(2*np.pi)
        whole = idx.astype(int)
        frac = idx - whole
        shifted[:,t] = np.roll(np.abs(1-frac)*x[:,t], whole)
        shifted[:,t] += np.roll(np.abs(frac)*x[:,t], int(whole+np.sign(frac)))

    return shifted