# Utils for getting frames that are related in the slice dimension, be it sharing a slice or across slices
from typing import List
import numpy as np
from .imparams import ImParams

def framelist_by_slice(im_params : ImParams, color_channel : int = None, upper_bound = None, slice_idx : int = None) -> list[list[int]]:
    """
    List of lists, each sublist containing the frames indices that share a z slice
    
    If slice_idx is not None, returns a single list corresponding to that slice.

    Defaults to using the lowest-numbered color channel
    """
    
    n_slices = im_params['NUM_SLICES']
    fps = im_params['FRAMES_PER_SLICE']
    colors = im_params['COLORS']
    n_frames = im_params['NUM_FRAMES']

    if isinstance(colors, list):
        n_colors = len(colors)
    else:
        n_colors = 1

    frames_per_volume = n_slices * fps * n_colors

    n_frames -= n_frames%frames_per_volume # ensures this only goes up to full volumes.

    if (color_channel is None) and (n_colors == 1):
        color_channel = 0
    if (color_channel is None) and (n_colors > 1):
        color_channel = min(colors) - 1 # MATLAB idx is 1-based

    frame_list = []

    all_frames = np.arange( n_frames - (n_frames % frames_per_volume))
    all_frames = all_frames.reshape(n_frames//frames_per_volume, n_slices * fps, n_colors)

    if slice_idx is None:
        for slice_num in range(n_slices):
            frame_list.append(all_frames[:,(slice_num*fps):((slice_num+1)*fps),color_channel].flatten().tolist()) # list of lists
    else:
        if not (isinstance(slice_idx, int)):
            slice_idx = int(slice_idx)
        frame_list = all_frames[:,(slice_idx*fps):((slice_idx+1)*fps),color_channel].flatten().tolist() # just a list
    return frame_list

def framelist_by_timepoint(im_params : ImParams, color_channel : int, upper_bound = None, slice_idx : int = None)->list[list[int]]:
    """
    If slice_idx is None:
    list of lists, each containing frames that share a timepoint, i.e. same stack but different slices or colors
    
    If slice_idx is int:
    Returns all frames corresponding to a specific slice
    """
    
    n_slices = im_params['NUM_SLICES']
    fps = im_params['FRAMES_PER_SLICE']
    colors = im_params['COLORS']
    n_frames = im_params['NUM_FRAMES']

    if isinstance(colors, list):
        n_colors = len(colors)
    else:
        n_colors = 1

    frames_per_volume = n_slices * fps * n_colors

    n_frames -= n_frames%frames_per_volume # ensures this only goes up to full volumes.

    if (color_channel is None) and (n_colors == 1):
        color_channel = 0
    if (color_channel is None) and (n_colors > 1):
        color_channel = min(colors) - 1 # MATLAB idx is 1-based

    all_frames = np.arange( n_frames - (n_frames % frames_per_volume))
    all_frames = all_frames.reshape(int(n_frames/frames_per_volume), n_slices * fps, n_colors)
    if slice_idx is None:
        if upper_bound is None:
            return all_frames[:,:,color_channel].tolist()
        else:
            return [framenum for framenum in all_frames[:,:,color_channel].tolist() if framenum < upper_bound]
    else:
        if not (type(slice_idx) is int):
            slice_idx = int(slice_idx)
        return all_frames[:, (slice_idx*fps):((slice_idx+1)*fps), color_channel].flatten().tolist()

def framelist_by_color(im_params : ImParams, color_channel : int, upper_bound = None)->list:
    "List of all frames that share a color, regardless of slice. Color channel is indexed from 0!"

    colors = im_params['COLORS']
    n_frames = im_params['NUM_FRAMES'] - 1
    
    if isinstance(colors, list):
        n_colors = len(colors)
    else:
        n_colors = 1

    if color_channel >= n_colors:
        raise ValueError("Color channel specified larger than number of colors acquired. Color channel is indexed from 0!")

    if upper_bound is None:
        return list(range(color_channel, n_frames, n_colors))
    else:
        return list(range(color_channel, upper_bound, n_colors))
