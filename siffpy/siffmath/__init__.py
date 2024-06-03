from typing import List, Tuple
import inspect
import textwrap

from scipy.signal import correlate, correlation_lags
import numpy as np

from siffpy.siffmath.phase import phase_alignment_functions # noqa: F401
from siffpy.siffmath.phase.phase_estimates import estimate_phase # noqa: F401
from siffpy.siffmath.flim import FlimTrace # noqa: F401
from siffpy.siffmath.fluorescence import (
    FluorescenceTrace, dFoF, photon_counts, # noqa: F401
)
from siffpy.siffmath.utils import Timeseries # noqa: F401
import siffpy.siffmath.fluorescence as fluorescence

def fluorescence_fcns(print_docstrings : bool = True) -> List[str]:
    """
    List of public functions available from fluorescence
    submodule. Seems a little silly since I can just use
    the __all__ but this way I can also print the
    docstrings.
    """
    fcns = inspect.getmembers(
        fluorescence,
        lambda x: inspect.isfunction(x) and issubclass(inspect.signature(x).return_annotation, FluorescenceTrace)
    )

    print_string = ""

    for member_fcn_info in fcns:
        fcn_name = member_fcn_info[0]
        fcn_call = member_fcn_info[1]
        print_string += f"\033[1m{fcn_name}\033[0m\n\n"
        print_string += f"\t{fcn_name}{inspect.signature(fcn_call)}\n\n"
        print_string += textwrap.indent(str(inspect.getdoc(fcn_call)),"\t\t")
        print_string += "\n\n"

    if print_docstrings:
        print(print_string)
    else:
        return fcns
    
def correlate_series(
        x : np.ndarray,
        y : np.ndarray,
        dt : float = 1.0
    )->Tuple[np.ndarray, np.ndarray]:
    """
    Correlate two time-series and return the time-delayed
    Pearson correlation
    coefficients and lags (in units of samples if `dt` is
    not specified). Performed by z-scoring the two timeseries
    and running `scipy.correlate` on that.

    Parameters
    ----------
    x : np.ndarray
        The first time series to correlate

    y : np.ndarray
        The second time series to correlate

    dt : float
        The time step between samples. Default is 1.0

    Returns
    -------
    (lags, corr) : Tuple[np.ndarray, np.ndarray]
        The lags and correlation coefficients. Dimensions
        of `corr` will be `len(x) + len(y) - 1`

    Example
    -------
    ```python 
    import numpy as np
    from siffpy.siffmath import correlate_series

    x = np.random.randn(100)
    y = np.random.randn(100)
    dt = 0.01
    lags, corr = corr_series(x, y, dt = dt)
    ```
    """
    x = x - np.mean(x)
    y = y - np.mean(y)

    corrd = correlate(x, y, mode = 'full') / (np.std(x) * np.std(y) * len(x))
    lags = correlation_lags(len(x), len(y), mode = 'full') * dt
    return lags, corrd