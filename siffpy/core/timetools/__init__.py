import numpy as np
import logging

from ..io import FrameMetaData, frame_metadata_to_dict
from ..utils import ImParams

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
    num_slices = im_params['NUM_SLICES']
    
    num_colors = 1
    if im_params is list:
        num_colors = len(im_params['COLORS'])

    fps = im_params['FRAMES_PER_SLICE']
    
    timestep_size = num_slices*num_colors*fps # how many frames constitute a timepoint
    
    # figure out the start and stop points we're analyzing.
    frame_start = timepoint_start * timestep_size
    
    if timepoint_end is None:
        frame_end = im_params['NUM_FRAMES']
    else:
        if timepoint_end > im_params['NUM_FRAMES']//timestep_size:
            logging.warning(
                "\ntimepoint_end greater than total number of frames.\n"+
                "Using maximum number of complete timepoints in image instead.\n"
            )
            timepoint_end = im_params['NUM_FRAMES']//timestep_size
        
        frame_end = timepoint_end * timestep_size

    # now convert to a list of all the frames whose metadata we want
    framelist = [frame for frame in range(frame_start, frame_end) 
        if (((frame-frame_start) % timestep_size) == (num_colors*reference_z))
    ]
    
    return np.array([frame['frameTimestamps_sec']
        for frame in frame_metadata_to_dict(frame_metadata)
    ])