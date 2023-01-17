# Code for ROI extraction from the noduli after manual input

from siffpy.siffplot.roi_protocols import rois
import holoviews as hv

def hemispheres(reference_frames : list, polygon_source : dict, *args, slice_idx : int = None, **kwargs) -> rois.Blobs:
    """
    Just takes the ROIs in the left and right hemispheres. Returns them as a Blobs ROI which is basically
    the same as two polygons. For now, it just takes the two largest ones! TODO: At least check left vs. right,
    and order them consistently
    """
    using_holoviews = True
    if type(polygon_source) is dict: # use holoviews
        annotation_dict = polygon_source
    else:
        using_holoviews = False
        raise NotImplementedError("Noduli hemispheres not implemented without holoviews yet")

    slice_idx = None
    if 'slice_idx' in kwargs:
        if isinstance(kwargs['slice_idx'], int):
            slice_idx = kwargs['slice_idx']
        del kwargs['slice_idx']

    if using_holoviews:    
        if len(annotation_dict[slice_idx]['annotator'].annotated.data) <2:
            raise ValueError("Fewer than two ROIs provided")

        poly_list = rois.get_largest_polygon_hv(annotation_dict, slice_idx = slice_idx, n_polygons = 2)
        polygons_combined = hv.Polygons(
            [
                {
                    'x' : polygon.data[0]['x'],
                    'y' : polygon.data[0]['y']
                }
                for polygon in poly_list
            ]
        )

    return rois.Blobs(polygons_combined, slice_idx)

def dummy_method(*args, **kwargs):
    print("I'm just a placeholder!")