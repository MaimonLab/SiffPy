""" Methods for computing F0 """

from typing import Optional
from logging import warning

import numpy as np

from siffpy.siffmath.utils.types import ImageArray

def nth_percentile(
    rois : 'ImageArray',
    n : float,
    rolling_window : Optional[int] = None,
    ignore_zeros : bool = False,
):
    """
    Roi-wise nth percentile value

    Parameters
    ----------
    rois : np.ndarray
        A 2D array of shape n_roi, n_frames

    n : float
        The percentile to compute (0 to 100)

    rolling_window : int (optional)

        If provided, the 5th percentile is computed using a rolling window of this size.

    ignore_zeros : bool

        If True, then zeros are ignored when computing the percentile.
    
    If rolling_window is None, returns a n_roi, 1 array.

    If rolling_window is not None, returns a n_roi, n_frames array for broadcasting purposes...
    Maybe this is dumb
    """

    if rolling_window is None:
        sorted_array = np.sort(rois,axis=-1)
        if ignore_zeros:
            return np.array([
                roi[roi!=0][n*len(roi[roi!=0])//100]
                for roi in sorted_array
            ])

        return sorted_array.take(n*sorted_array.shape[-1]//100, axis=-1)
    
    if rolling_window > 1000:
        warning("Large rolling window size. This may be slow. "+
                "Remind me to implement a faster version of this at some point."
        )
    return compute_rolling_baseline(
        rois,
        rolling_window,
        percentile = n/100,
        ignore_zeros = ignore_zeros,
    )


def fifth_percentile(
        rois : 'ImageArray',
        rolling_window : Optional[int] = None,
        ignore_zeros : bool = False,
    ) -> np.ndarray:
    """
    Roi-wise 5th percentile value
    
    Rolling_window : int (optional)

        If provided, the 5th percentile is computed using a rolling window of this size.
    
    If rolling_window is None, returns a n_roi, 1 array.

    If rolling_window is not None, returns a n_roi, n_frames array for broadcasting purposes...
    Maybe this is dumb
    """
    return nth_percentile(rois, 5, rolling_window, ignore_zeros=ignore_zeros)

def roi_mean(rois : 'ImageArray') -> np.ndarray:
    """ Takes the mean within each ROI """
    return np.mean(rois,axis=1)

def compute_rolling_baseline(
    f_array : 'ImageArray',
    width : int,
    percentile : float = 0.05,
    ignore_zeros : bool = False,
):  
    frac_zeros = 0
    # Suboptimal....
    if ignore_zeros:
        frac_zeros += np.sum(f_array==0)/f_array.size
    from scipy.ndimage import percentile_filter
    #""" WARNING: SLOW FOR LARGE WINDOWS. Should do this better."""
    size = (1,int(width//2)) if f_array.ndim > 1 else int(width//2)
    return percentile_filter(f_array, (frac_zeros + percentile)*100, size=size)