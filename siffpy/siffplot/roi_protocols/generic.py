# Code for ROI extraction from the fan-shaped body after manual input

from siffpy.siffplot.roi_protocols import rois
from siffpy.siffplot.roi_protocols.utils import PolygonSource

def outline_roi(reference_frames : list, polygon_source : PolygonSource, *args, slice_idx : int = None, **kwargs)-> rois.ROI:
    """
    Takes the largest ROI and assumes it's the outline of the ROI of interest.

    Parameters
    ----------

    reference_frames : list of numpy arrays

        The siffreader reference frames to overlay

    polygon_source : PolygonSource

        Backend-invariant polygon source representation

    slice_idx : None, int, or list of ints (optional)

        Which slice or slices to extract an ROI for. If None, will take the ROI
        corresponding to the largest polygon across all slices.
    """
    
    largest_polygon, slice_idx, _ = polygon_source.get_largest_polygon(slice_idx = slice_idx)
    source_image = polygon_source.source_image(slice_idx)
    return rois.ROI(
        largest_polygon,
        slice_idx = slice_idx,
        image = source_image
    )

def dummy_method(*args, **kwargs):
    print("I'm just a placeholder!")