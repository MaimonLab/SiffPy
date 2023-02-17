import numpy as np
from scipy.ndimage.filters import uniform_filter1d

import logging

from siffpy.core.io import FrameMetaData, frame_metadata_to_dict
from siffpy.core.utils import ImParams

SEC_TO_NANO = 1e9
NANO_TO_SEC = 1e-9

def epoch_to_frame_time(epoch_time : int, frame_meta : FrameMetaData)->float:
    """ Converts epoch time to frame time for this experiment (returned in seconds) """
    frame_zero_time = frame_meta['frameTimestamps_sec'] # in seconds
    epoch_zero_time = frame_meta['epoch'] # in nanoseconds
    offset = frame_zero_time * SEC_TO_NANO - epoch_zero_time
    return (epoch_time + offset)/NANO_TO_SEC

def metadata_dicts_to_time(dicts : list[dict], reference : str = "experiment")->np.ndarray:
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

def to_t_axis(
        frame_metadata : list[FrameMetaData],
        im_params : ImParams,
        timepoint_start : int = 0,
        timepoint_end : int = None,
        reference_z : int = 0,
    )->np.ndarray:
    
    if timepoint_end > im_params.num_timepoints:
        logging.warning(
            "\ntimepoint_end greater than total number of frames.\n"+
            "Using maximum number of complete timepoints in image instead.\n"
        )
        timepoint_end = im_params.num_timepoints
        
    # now convert to a list of all the frames whose metadata we want
    framelist = [
        im_params.framelist_by_slice(
            im_params.colors[0], # colors are simultaneous
            upper_bound = timepoint_end,
            slice_idx = reference_z
        )
    ]
    
    return np.array([frame['frameTimestamps_sec']
        for frame in frame_metadata_to_dict(frame_metadata)
    ])

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