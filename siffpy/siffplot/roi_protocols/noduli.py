# Code for ROI extraction from the noduli after manual input

from .rois import *

def hemispheres(reference_frames : list, annotation_dict : dict, *args, **kwargs) -> hv.element.path.Polygons:
    """
    Just takes the ROIs in the left and right hemispheres.
    """

    slice_idx = None
    if 'slice_idx' in kwargs:
        if isinstance(kwargs['slice_idx'], int):
            slice_idx = kwargs['slice_idx']
        del kwargs['slice_idx']
    
    largest_polygon, slice_idx, roi_idx = get_largest_polygon(annotation_dict, slice_idx = slice_idx)

    if len(annotation_dict[slice_idx]['annotator'].annotated.data) <2:
        raise ValueError("Fewer than two ROIs provided")

    raise NotImplementedError("Not finished method")

    

def dummy_method(*args, **kwargs):
    print("I'm just a placeholder!")