# Dedicated code for data that is purely fluorescence analysis

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

def dFoF(roi : np.ndarray, normalized : bool = False, Fo = lambda x: np.mean(x,axis=1))->np.ndarray:
    """
    
    Takes a numpy array and returns a dF/F trace across the rows -- i.e. each row is normalized independently
    of the others. Returns a version of the function (F - F0)/F0, where F0 is computed as below

    normalized : bool (optional)

        Compresses the response of each row to approximately the range 0 - 1 (uses the 5th and 95th percentiles).
        Default is False

    Fo : callable or np.ndarray (optional)

        How to determine the F0 term for a given row. If Fo is callable, the function is applied to the
        roi numpy array directly (i.e. it's NOT a function that operates on one row). Can also provide
        just a number or an array of numbers.
    
    """
    if not type(roi) == np.ndarray:
        roi = np.array(roi)
    
    if callable(Fo):
        F0 = Fo(roi)
    elif type(Fo) is np.ndarray or float:
        F0 = Fo
    else:
        try:
            np.array(Fo)
        except:
            raise TypeError(f"Keyword argument Fo is not of type float, a numpy array, or a callable, nor can it be cast to such.")

    unnormalized = ((roi.T - F0)/F0).T
    
    if normalized:
        sorted_vals = np.sort(unnormalized,axis=1)
        min_val = sorted_vals[:,sorted_vals.shape[-1]//20]
        max_val = sorted_vals[:,int(sorted_vals.shape[-1]*(1.0-1.0/20))]
        return ((unnormalized.T - min_val)/(max_val - min_val)).T
    
    return unnormalized

def roi_masked_fluorescence_numpy(fluorescence : np.ndarray, rois : list[np.ndarray]):
    raise NotImplementedError()

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
        raise NotImplementedError()
else:
    def roi_masked_fluorescence(fluorescence : np.ndarray, rois: list[np.ndarray]) -> list[np.ndarray]:
        """
        As above, but does not rely on Holoviews for the rois. Expects just lists of
        numpy arrays for the ROI masks.
        """
        return roi_masked_fluorescence_numpy(fluorescence, rois)
