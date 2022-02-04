"""
Dedicated code for data that is purely fluorescence analysis
"""
import numpy as np
import logging
holoviews : bool = False
try:
    from ..siffplot.roi_protocols.rois.roi import ROI
    holoviews = True
except ImportError:
    logging.warn("""

    WARNING:

    ROI functionality depends on Holoviews for polygons.

    Other functions will operate just fine without it, so
    I'm still allowing you to perform some imports, but
    this is a warning that things might not work.
    
    """
    )

__all__ = [
    "dFoF",
    "roi_masked_fluorescence"
]

def dFoF(fluorescence : np.ndarray, *args, normalized : bool = False, Fo = lambda x: np.mean(x,axis=1), **kwargs)->np.ndarray:
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
    if not type(fluorescence) == np.ndarray:
        fluorescence = np.array(fluorescence)
    
    if callable(Fo):
        F0 = Fo(fluorescence, *args, **kwargs)
    elif type(Fo) is np.ndarray or float:
        F0 = Fo
    else:
        try:
            np.array(Fo)
        except:
            raise TypeError(f"Keyword argument Fo is not of type float, a numpy array, or a callable, nor can it be cast to such.")

    unnormalized = ((fluorescence.T - F0)/F0).T
    
    if normalized:
        sorted_vals = np.sort(unnormalized,axis=1)
        min_val = sorted_vals[:,sorted_vals.shape[-1]//20]
        max_val = sorted_vals[:,int(sorted_vals.shape[-1]*(1.0-1.0/20))]
        return ((unnormalized.T - min_val)/(max_val - min_val)).T
    
    return unnormalized

def roi_masked_fluorescence_numpy(frames : np.ndarray, rois : list[np.ndarray]):
    """
    Takes an array of frames organized as a k-dimensional numpy array with the
    last three dimensions being ('time', 'y', 'x') and converts them into an k-2
    dimensional array, with the final two dimensions of 'frames' compressed against
    the masks in rois.
    """
    rois = np.array(rois)
    return np.sum(
            np.tensordot(
                frames,
                rois,
                axes=((-1,-2),(-1,-2))
            ),
            axis = (-2,-1)
        )

if holoviews:
    def roi_masked_fluorescence(fluorescence : np.ndarray, rois : list[ROI]) -> list[np.ndarray]:
        """
        Returns a list of numpy arrays, each corresponding to the masked version of the input image
        series (itself a numpy array or list of numpy arrays)

        Arguments
        ---------

        Returns
        -------
        
        """
        return roi_masked_fluorescence_numpy(fluorescence, [roi.mask() for roi in rois])
else:
    def roi_masked_fluorescence(fluorescence : np.ndarray, rois: list[np.ndarray]) -> list[np.ndarray]:
        """
        As above, but does not rely on Holoviews for the rois. Expects just lists of
        numpy arrays for the ROI masks.
        """
        return roi_masked_fluorescence_numpy(fluorescence, rois)
