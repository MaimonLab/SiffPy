# Utils for getting frames that are related in the slice dimension, be it sharing a slice or across slices
import numpy as np
from .imparams import ImParams

def framelist_by_slice(im_params : ImParams, color_channel : int) -> list[list[int]]:
    """ List of lists, each sublist containing the frames indices that share a z slice """
    
    n_slices = im_params['NUM_SLICES']
    fps = im_params['FRAMES_PER_SLICE']
    colors = im_params['COLORS']
    n_frames = im_params['NUM_FRAMES'] - 1

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
    for slice_idx in range(n_slices):
        slice_offset = slice_idx * fps * n_colors # frames into volume before this slice begins
        slice_offset += color_channel

        # HACKY CORRECTION!!!
        slice_offset -= n_colors*(fps > 1) # shift every frame index back by one frame if there are multiple frames perslice

        slice_list = [] # all frames in this z plane
        for frame_num in range(fps):
            fn = frame_num * n_colors # 1 for each frame in the same slice
            slice_list += list(range(slice_offset+fn, n_frames, frames_per_volume))
            slice_list = list(filter(lambda x: x>=0,slice_list)) # only needed for that -1 issue

        frame_list.append(slice_list)        
    
    return [frame_list[n] for n in range(len(frame_list))]

def framelist_by_timepoint(im_params : ImParams, color_channel : int)->list[list[int]]:
    """ List of lists, each containing frames that share a timepoint"""
    
    n_slices = im_params['NUM_SLICES']
    fps = im_params['FRAMES_PER_SLICE']
    colors = im_params['COLORS']
    n_frames = im_params['NUM_FRAMES'] - 1

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
    return all_frames[:,:,color_channel].tolist()

def framelist_by_color(im_params : ImParams, color_channel : int)->list:
    "List of all frames that share a color, regardless of slice"

    colors = im_params['COLORS']
    n_frames = im_params['NUM_FRAMES'] - 1
    
    if isinstance(colors, list):
        n_colors = len(colors)
    else:
        n_colors = 1

    if color_channel > n_colors:
        raise ValueError("Color channel specified larger than number of colors acquired")

    return list(range(color_channel, n_frames, n_colors))
