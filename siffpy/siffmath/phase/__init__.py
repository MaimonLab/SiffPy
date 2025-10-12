import inspect
import textwrap
from typing import List, Optional, Tuple, Any

import numpy as np

from siffpy.siffmath.phase.traces import PhaseTrace
import siffpy.siffmath.phase.phase_estimates as phase_estimates
from siffpy.siffmath.utils.types import PhaseTraceLike

def phase_alignment_functions(print_docstrings : bool = True)->Optional[List[Tuple[str, Any]]]:
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
    
def phase_shift(
        x : np.ndarray,
        shift : PhaseTraceLike,
        shift_axis : int = 0,
        time_axis : int = -1
    )->np.ndarray:
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

    time_axis : int, optional
        The axis of `x` that is the time axis to iterate along, by default -1

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
    if len(shift) != x.shape[time_axis]:
        raise ValueError(
            f"`shift` must have the same number of elements as the `time_axis` dimension of `x` \
            input was of shape {x.shape}, `time_axis` was {time_axis} \
            and `shift` was of length {len(shift)}."
        )
    if np.issubdtype(shift.dtype, np.floating):
        shift = np.angle(np.exp(1j*shift))
    if isinstance(shift, PhaseTrace):
        shift = np.angle(shift)

    # Fraction of the column dimension to shift by
    shift = np.mod(shift, 2 * np.pi) / (2*np.pi)

    shifted = np.zeros_like(x)
    n_cols = x.shape[shift_axis]
    for t in range(len(shift)):
        idx = n_cols * shift[t]
        whole = idx.astype(int)
        frac = idx - whole # always positive
        
        # Not sure how to vectorize this part
        this_row = np.take(x, t, axis = time_axis)
        new_row = np.roll((1-frac)*this_row, whole, axis = shift_axis)
        new_row += np.roll(frac*this_row, whole+1, axis = shift_axis)

        if time_axis == -1:
            indices = np.s_[
                (slice(None),) * (shifted.ndim - 1) + (t,)
            ]
        else:
            indices = np.s_[
                (slice(None),) * time_axis + (t,) + (slice(None),) * (shifted.ndim - time_axis - 1)
            ]

        shifted[indices] = new_row


    return shifted