"""
Dedicated code for data that is purely fluorescence analysis
"""
from typing import Callable, Union, TYPE_CHECKING
import numpy as np

from siffpy.siffmath.fluorescence.traces import FluorescenceTrace, FluorescenceVector # noqa: F401
from siffpy.siffmath.fluorescence.baseline_methods import fifth_percentile

if TYPE_CHECKING:
    from siffpy.siffmath.utils.types import FluorescenceArrayLike
        
def photon_counts(fluorescence : 'FluorescenceArrayLike', *args, **kwargs)->FluorescenceTrace:
    """ Simply returns raw photon counts. This is just the array that's passed in, wrapped in a FluorescenceTrace. """
    return FluorescenceTrace(fluorescence, F = fluorescence, method = 'Photon counts', F0 = 0)

def dFoF(
        fluorescence : 'FluorescenceArrayLike',
        *args,
        normalized : bool = False,
        Fo : Union[np.ndarray, float, Callable[[np.ndarray], Union[np.ndarray, int, float]]] = fifth_percentile,
        **kwargs
    )->FluorescenceTrace:
    """
    
    Takes a numpy array and returns a dF/F0 trace across the rows -- i.e. each row is normalized independently
    of the others. Returns a version of the function (F - F0)/F0, where F0 is computed as below

    fluorescence : np.ndarray

        The data constituting the F in dF/F0

    normalized : bool (optional)

        Compresses the response of each row to approximately the range 0 - 1 (uses the 5th and 95th percentiles).
        Default is False

    Fo : callable or np.ndarray (optional)

        How to determine the F0 term for a given row. If Fo is callable, the function is applied to the
        roi numpy array directly (i.e. it's NOT a function that operates on only one row at a time). 
        Can also provide just a number or an array of numbers.

    Passes additional args and kwargs to the Fo function, if those args and kwargs are provided.
    
    """
    if not isinstance(fluorescence,np.ndarray):
        fluorescence = np.array(fluorescence)
    fluorescence = np.atleast_2d(fluorescence)
    
    #info_string = ""
    if callable(Fo):
        F0 = Fo(fluorescence, *args, **kwargs)
        #inspect.signature(Fo).
    else:
        try:
            F0 = np.array(Fo).astype(float)
        except TypeError:
            raise TypeError(
                "Keyword argument Fo is not of type float, a numpy array, "
                 + "or a callable, nor can it be cast to such."
            )
    
    F0 = np.atleast_2d(F0)
    # Code smell, BIG TIME. Quick fix for the way numpy does the atleast_2d thing...
    if F0.shape[-1] == fluorescence.shape[0]:
        F0 = F0.T
    
    df_trace = ((fluorescence.astype(float) - F0.astype(float))/F0.astype(float))
    
    max_val = None
    min_val = None

    if normalized:
        sorted_vals = np.sort(df_trace,axis=1)
        min_val = sorted_vals[:,sorted_vals.shape[-1]//20]
        max_val = sorted_vals[:,int(sorted_vals.shape[-1]*(1.0-1.0/20))]
        df_trace = ((df_trace.T - min_val)/(max_val - min_val)).T

    df_trace = df_trace.squeeze()

    args_str = "args :" + ", ".join([str(arg) for arg in args])
    kwargs_str = "kwargs :" + ", ".join([f"{key} = {value}" for key, value in kwargs.items()])

    return FluorescenceTrace(
        df_trace,
        normalized = normalized,
        method = 'dF/F', 
        F = fluorescence,
        F0 = F0,
        max_val = max_val,
        min_val = min_val,
        info_string = f"Computed dFoF trace with {args_str} and {kwargs_str}"
    )
