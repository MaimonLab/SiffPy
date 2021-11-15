# Code for ROI extraction from the fan-shaped body after manual input

from . import rois

def outline_fan(reference_frames : list, annotation_dict : dict, *args, **kwargs):
    """
    Takes the largest ROI and assumes it's the outline of the fan-shaped body
    """
    slice_idx = None
    if 'slice_idx' in kwargs:
        if isinstance(kwargs['slice_idx'], int):
            slice_idx = kwargs['slice_idx']
        del kwargs['slice_idx']
    
    largest_polygon, slice_idx, roi_idx = rois.get_largest_polygon(annotation_dict, slice_idx = slice_idx)
    
    return rois.Fan(
        largest_polygon,
        slice_idx = slice_idx,
        image = rois.annotation_dict_to_numpy(annotation_dict,slice_idx)
    )