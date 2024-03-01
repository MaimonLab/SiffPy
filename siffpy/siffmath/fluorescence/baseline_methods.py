""" Methods for computing F0 """

from typing import Optional, Any
from logging import warning

import numpy as np

from siffpy.siffmath.utils.types import ImageArray

def nth_percentile(
    rois : 'ImageArray',
    n : float,
    rolling_window : Optional[int] = None
):
    """
    Roi-wise nth percentile value

    Parameters
    ----------
    rois : np.ndarray
        A 2D array of shape n_roi, n_frames

    n : float
        The percentile to compute (0 to 100)

    Rolling_window : int (optional)

        If provided, the 5th percentile is computed using a rolling window of this size.
    
    If rolling_window is None, returns a n_roi, 1 array.

    If rolling_window is not None, returns a n_roi, n_frames array for broadcasting purposes...
    Maybe this is dumb
    """

    if rolling_window is None:
        sorted_array = np.sort(rois,axis=-1)
        return sorted_array.take( int(sorted_array.shape[-1] / (100/n)) ,axis=-1)
    
    if rolling_window > 1000:
        warning("Large rolling window size. This may be slow. "+
                "Remind me to implement a faster version of this at some point."
        )
    return compute_rolling_baseline(
        rois,
        rolling_window,
        percentile = n/100
    )


def fifth_percentile(
        rois : 'ImageArray',
        rolling_window : Optional[int] = None
    ) -> np.ndarray:
    """
    Roi-wise 5th percentile value
    
    Rolling_window : int (optional)

        If provided, the 5th percentile is computed using a rolling window of this size.
    
    If rolling_window is None, returns a n_roi, 1 array.

    If rolling_window is not None, returns a n_roi, n_frames array for broadcasting purposes...
    Maybe this is dumb
    """
    return nth_percentile(rois, 5, rolling_window)

def roi_mean(rois : 'ImageArray') -> np.ndarray:
    """ Takes the mean within each ROI """
    return np.mean(rois,axis=1)

def compute_rolling_baseline(
    f_array : 'ImageArray',
    width : int,
    percentile : float = 0.05
):
    from scipy.ndimage import percentile_filter
    #""" WARNING: SLOW FOR LARGE WINDOWS. Should do this better."""
    size = (1,int(width//2)) if f_array.ndim > 1 else int(width//2)
    return percentile_filter(f_array, percentile*100, size=size)
    # nth_p = int(width*percentile)
    # pure_idxs = np.array(list(range(len(f_array))))
    # return np.array(
    #     [
    #         f_array[sorted_f_idxs[np.abs(sorted_f_idxs - idx) <= width][nth_p]]
    #         for idx in pure_idxs
    #     ]
    # )
