import numpy as np

def fifth_percentile(rois : np.ndarray) -> np.ndarray:
    """ Roi-wise 5th percentile value """
    sorted_array = np.sort(rois,axis=-1)
    return sorted_array.take( sorted_array.shape[-1] // 20 ,axis=-1)

def roi_mean(rois : np.ndarray) -> np.ndarray:
    """ Takes the mean within each ROI """
    return np.mean(rois,axis=1)