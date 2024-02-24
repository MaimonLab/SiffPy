from typing import List, Dict

import numpy as np
from scipy.ndimage import uniform_filter1d

from siffpy.core.io import FrameMetaData, frame_metadata_to_dict

SEC_TO_NANO = 1e9
NANO_TO_SEC = 1e-9

def epoch_to_frame_time(epoch_time : int, frame_meta : FrameMetaData)->float:
    """ Converts epoch time to frame time for this experiment (returned in seconds) """
    frame_zero_time : float = frame_meta['frameTimestamps_sec'] # in seconds
    epoch_zero_time : int = frame_meta['epoch'] # in nanoseconds
    offset = frame_zero_time * SEC_TO_NANO - epoch_zero_time
    return (epoch_time + offset)/NANO_TO_SEC

def frame_time_to_epoch(frame_time : float, frame_meta : FrameMetaData)->int:
    """ Converts frame time to epoch time for this experiment (returned in nanoseconds) """
    frame_zero_time : float = frame_meta['frameTimestamps_sec'] # in seconds
    epoch_zero_time : int = frame_meta['epoch'] # in nanoseconds
    offset = frame_zero_time * SEC_TO_NANO - epoch_zero_time
    return int(frame_time * SEC_TO_NANO - offset)

def metadata_dicts_to_time(dicts : List[Dict], reference : str = "experiment")->np.ndarray:
    """ 
    Takes an iterable of metadata dictionaries output by
    get_frame_metadata and converts them into a single
    numpy array
    """
    reference = reference.lower() # case insensitive

    if reference == "epoch":
        return np.array([frame['epoch'] # WARNING, OLD VERSIONS USED SECONDS NOT NANOSECONDS 
            for frame in frame_metadata_to_dict(dicts)
        ])
    
    if reference == "experiment":
        return np.array([frame['frameTimestamps_sec']
            for frame in frame_metadata_to_dict(dicts)
        ])
    else:
        ValueError("Reference argument not a valid parameter (must be 'epoch' or 'experiment')")

def rolling_avg(trace : np.ndarray, time_axis : np.ndarray, window_length : float)->np.ndarray:
    """
    Computes a rolling average of a 1d timeseries

    Arguments
    ---------
    
    trace : np.ndarray

        An axis of points to take the rolling average of

    time_axis : np.ndarray

        An axis of the same shape as trace with corresponding timepoints

    window_length : float

        How long the rolling average should be taken over, in the same units as time_axis

    Returns
    -------

    avgd : np.ndarray

    The rolling average of the input array trace
    """
    dt = np.mean(np.diff(time_axis))
    bin_width = max(int(window_length//dt),1)

    return uniform_filter1d(trace, bin_width,)