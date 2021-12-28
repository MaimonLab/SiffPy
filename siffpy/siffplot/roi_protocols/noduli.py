# Code for ROI extraction from the noduli after manual input

from . import rois
import holoviews as hv

def hemispheres(reference_frames : list, polygon_source : dict, *args, **kwargs) -> hv.element.path.Polygons:
    """
    Just takes the ROIs in the left and right hemispheres. Returns them as a Blob ROI which is basically
    the same as a polygon.
    """
    if type(polygon_source) is dict: # use holoviews
        annotation_dict = polygon_source

    slice_idx = None
    if 'slice_idx' in kwargs:
        if isinstance(kwargs['slice_idx'], int):
            slice_idx = kwargs['slice_idx']
        del kwargs['slice_idx']
    
    if len(annotation_dict[slice_idx]['annotator'].annotated.data) <2:
        raise ValueError("Fewer than two ROIs provided")

    poly_list = rois.get_largest_polygon_hv(annotation_dict, slice_idx = slice_idx, n_polygons = 2)


    raise NotImplementedError("Not finished method")

    

def dummy_method(*args, **kwargs):
    print("I'm just a placeholder!")