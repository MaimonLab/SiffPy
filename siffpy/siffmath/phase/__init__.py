import inspect
import textwrap
from typing import List, Optional, Tuple, Any

import numpy as np
from scipy.interpolate import interp1d

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
        x : np.ndarray[Any, Any],
        shift : PhaseTraceLike,
        shift_axis : int = 0,
        time_axis : int = -1,
        angle_coords : Optional[np.ndarray[Any, np.dtype[np.floating]]] = None,
    )->np.ndarray:
    """
    Shifts the phase of a vector time series by the phase requested.

    Interpolates for phases that do not evenly divide the number of bins,
    (e.g. if there are 4 bins and the phase is 60 degrees, 2/3 of the mass
    of the 0th bin will go into the 90 degree bin, and 1/3 will go into the 0
    degree bin).

    ## Parameters

    - `x` : np.ndarray
        The vector time series to shift the phase of. Must be of shape
        (n_bins, n_time)

    - `phase` : PhaseTraceLike
        Accepts either an array of angles (does not need to be wrapped)
        or a `PhaseTrace` object.

    - `time_axis` : int, optional
        The axis of `x` that is the time axis to iterate along, by default -1

    - `angle_coords` : Optional[np.ndarray[Any, np.dtype[np.floating]]], optional
        If provided, the coordinates of the angle bins. This is useful, for example,
        if the angle bins are not evenly spaced, as in the EPGS of the protocerebral bridge,
        or if there are duplicates as in the Δ7s of the bridge. If `None`, the function
        will assume evenly distributed angle bins of width 2π/`x.shape[shift_axis]`
        starting from -π.

    ## Returns
    
    np.ndarray
        The phase shifted vector time series. Will be of same shape as `x`
        but rotated so that the `phase` is centered at the 0 degree bin, i.e.
        the `center` column if `angle_coords` is not provided.

    ## Example

    ```python

    import numpy as np
    from siffpy.siffmath.phase import phase_shift

    x = np.random.randn(16, 12000)
    d_phase = np.random.randn(12000) + 0.01
    phase = np.angle(np.exp(1j*np.cumsum(d_phase)))
    shifted = phase_shift(x, phase)

    ```

    ## Note

    This operation _will_ produce a sinusoid for identically and independently distributed
    Gaussian random variables! This is provable! Think of it this way: the phase operation
    will have a tendency to align to the position where multiple columns are positive, not
    the position where the greatest value is taken. So there will be a strong correlation
    between columns adjacent to the "phase null" column. This, for iid Gaussian rvs, is
    provably going to be a sinusoid that goes up and down by np.sqrt(np.pi/N) z-scores
    (e.g. ~0.5 z scores for 16 bins). For other random noise patterns, it produces
    a different distribution of values. But just make sure you do shuffle controls and
    don't take a sinusoid too seriously without doing your own due diligence!
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

    if angle_coords is not None:
        new_angles = np.linspace(-np.pi, np.pi, 2*x.shape[shift_axis], endpoint=False)
        x = interp1d(angle_coords, x,
            axis=shift_axis, kind = 'linear', bounds_error=False,
            fill_value="extrapolate"
        )(new_angles)

    shifted = np.zeros_like(x)
        
    # Fraction of the column dimension to shift by
    shift = np.mod(shift, 2 * np.pi) / (2*np.pi)

    n_cols = x.shape[shift_axis]
    
    # Create an array for indexing the columns
    bin_shape = [1] * x.ndim
    bin_shape[shift_axis] = n_cols
    bin_idx = np.arange(n_cols).reshape(bin_shape).astype(int)

    time_shape = [1] * x.ndim
    time_shape[time_axis] = len(shift)
    
    shift_vals = (shift * n_cols).reshape(time_shape)
    whole = shift_vals.astype(int)
    frac = shift_vals - whole # always positive between 0 and 1


    idx_low = (bin_idx - whole + 1) % n_cols
    idx_high = (bin_idx - whole) % n_cols

    shifted += np.take_along_axis(x, np.broadcast_to(idx_low, x.shape), axis=shift_axis) * (1 - frac)
    shifted += np.take_along_axis(x, np.broadcast_to(idx_high, x.shape), axis=shift_axis) * frac

    if angle_coords is not None:
        # Back to the old angle coordinates
        shifted = interp1d(new_angles, shifted,
            axis=shift_axis, kind = 'linear', bounds_error=False,
            fill_value="extrapolate"
        )(angle_coords)

    return shifted