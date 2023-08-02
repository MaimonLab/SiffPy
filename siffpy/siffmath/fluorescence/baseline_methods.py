""" Methods for computing F0 """

from typing import Optional, Any
from logging import warning

import numpy as np

from siffpy.siffmath.utils.types import ImageArray

def nth_percentile(
    rois : ImageArray,
    n : float,
    rolling_window : Optional[int] = None
):
    """
    Roi-wise nth percentile value
    Rolling_window : int (optional)

        If provided, the 5th percentile is computed using a rolling window of this size.
    
    If rolling_window is None, returns a n_roi, 1 array.

    If rolling_window is not None, returns a n_roi, n_frames array for broadcasting purposes...
    Maybe this is dumb
    """

    if rolling_window is None:
        sorted_array = np.sort(rois,axis=-1)
        return sorted_array.take( int(sorted_array.shape[-1] / (100/n)) ,axis=-1)
    
    sorted_array = np.argsort(rois,axis=-1)
    if rolling_window > 1000:
        warning("Large rolling window size. This may be slow. "+
                "Remind me to implement a faster version of this at some point."
        )
    if len(rois.shape) > 1:
        return np.array(
            [
                compute_rolling_baseline(
                    sub_roi,
                    sorted_array_sub,
                    rolling_window,
                    percentile = n/100,
                )
                for sub_roi, sorted_array_sub in zip(rois,sorted_array)
            ]
        )
    return compute_rolling_baseline(
        rois,
        sorted_array,
        rolling_window,
        percentile = n/100
    )


def fifth_percentile(
        rois : ImageArray,
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
    
    if rolling_window is None:
        sorted_array = np.sort(rois,axis=-1)
        return sorted_array.take( sorted_array.shape[-1] // 20 ,axis=-1)
    
    sorted_array = np.argsort(rois,axis=-1)
    if rolling_window > 1000:
        warning("Large rolling window size. This may be slow. "+
                "Remind me to implement a faster version of this at some point."
        )
    if len(rois.shape) > 1:
        return np.array(
            [
                compute_rolling_baseline(
                    sub_roi,
                    sorted_array_sub,
                    rolling_window,
                    percentile = 0.05,
                )
                for sub_roi, sorted_array_sub in zip(rois,sorted_array)
            ]
        )
    return compute_rolling_baseline(
        rois,
        sorted_array,
        rolling_window,
        percentile = 0.05
    )

def roi_mean(rois : ImageArray) -> np.ndarray:
    """ Takes the mean within each ROI """
    return np.mean(rois,axis=1)

def compute_rolling_baseline(
    f_array : ImageArray,
    sorted_f_idxs : np.ndarray[Any, np.dtype[np.uintc]],
    width : int,
    percentile : float = 0.05
):
    """ WARNING: SLOW FOR LARGE WINDOWS. Should do this better."""
    nth_p = int(width*percentile)
    pure_idxs = np.array(list(range(len(f_array))))
    return np.array(
        [
            f_array[sorted_f_idxs[np.abs(sorted_f_idxs - idx) <= width][nth_p]]
            for idx in pure_idxs
        ]
    )
