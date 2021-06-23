# Utils for getting frames that are related in the slice dimension, be it sharing a slice or across slices
import numpy as np

def framelist_by_slice(n_slices : int, fps : int, colors : list, color_channel : int, n_frames : int) -> list[list[int]]:
    """ List of lists, each sublist containing the frames indices that share a z slice """
    
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
        slice_list = [] # all frames in this z plane
        for frame_num in range(fps):
            fn = frame_num * n_colors # 1 for each frame in the same slice
            slice_list += list(range(slice_offset+fn, n_frames, frames_per_volume))

        frame_list.append(slice_list)        
    
    return [frame_list[n] for n in range(len(frame_list))]

def framelist_by_timepoint(n_slices : int, fps : int, colors : list, color_channel : int, n_frames : int)->list[list[int]]:
    """ List of lists, each containing frames that share a timepoint"""
    
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

