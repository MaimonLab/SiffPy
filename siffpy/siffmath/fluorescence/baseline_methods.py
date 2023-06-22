""" Methods for computing F0 """

from typing import Optional
from logging import warning

import numpy as np


def fifth_percentile(
        rois : np.ndarray,
        rolling_window : Optional[int] = None
    ) -> np.ndarray:
    """
    Roi-wise 5th percentile value
    Rolling_window : int (optional)

        If provided, the 5th percentile is computed using a rolling window of this size.
    """
    
    sorted_array = np.sort(rois,axis=-1)
    if rolling_window is None:
        return sorted_array.take( sorted_array.shape[-1] // 20 ,axis=-1)
    else:
        if rolling_window > 2000:
            warning(
                "Large rolling window size provided. This may be slow." +
                " Remind me to come back and reimplement this in a faster way"
            )
        return np.array(
            [
                compute_rolling_baseline(
                    sub_roi,
                    sorted_array_sub,
                    rolling_window
                )
                for sub_roi, sorted_array_sub in zip(rois,sorted_array)
                
            ] if len(rois.shape) == 2 else
            compute_rolling_baseline(
                rois,
                sorted_array,
                rolling_window
            )
        )

def roi_mean(rois : np.ndarray) -> np.ndarray:
    """ Takes the mean within each ROI """
    return np.mean(rois,axis=1)

def compute_rolling_baseline(
    f_array : np.ndarray,
    sorted_f_idxs : np.ndarray,
    width : int,
):
    """ WARNING: SLOW FOR LARGE WINDOWS. Should do this better."""
    fifth_p = int(width*0.05)
    pure_idxs = np.array(list(range(len(f_array))))
    return f_array[sorted_f_idxs[np.abs(sorted_f_idxs - pure_idxs) < width][fifth_p]]
