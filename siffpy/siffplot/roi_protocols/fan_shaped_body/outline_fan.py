# Code for ROI extraction from the fan-shaped body after manual input
import numpy as np

from siffpy.siffplot.roi_protocols import rois
from siffpy.siffplot.roi_protocols.roi_protocol import ROIProtocol
from siffpy.siffplot.roi_protocols.utils import PolygonSource

class OutlineFan(ROIProtocol):

    name = "Outline fan"
    base_roi_text = "Extract fan"

    def extract(
            self,
            reference_frames : np.ndarray,
            polygon_source : PolygonSource,
            slice_idx : int = 0,
    )->rois.Fan:
        return outline_fan(
            reference_frames,
            polygon_source,
            slice_idx=slice_idx,
        )

    def segment(self):
        raise NotImplementedError()

def outline_fan(reference_frames : list, polygon_source : PolygonSource, *args, slice_idx : int = None, **kwargs)-> rois.Fan:
    """
    Takes the largest ROI and assumes it's the outline of the fan-shaped body.

    Optionally looks for two lines to define the edges of the fan for a triangle-based
    segmentation of columns.

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
    ## Outline:
    #
    #   Just takes the largest polygon and assumes it's fan shaped.
    #
    #   Also looks to see if there's a pair of lines that can be used to guide
    #   segmentation, and if so adds that info to the Fan object produced.
    
    largest_polygon, slice_idx, _ = polygon_source.get_largest_polygon(slice_idx = slice_idx)
    source_image = polygon_source.source_image(slice_idx)
    fan_lines = None
    try:
        fan_lines = polygon_source.get_largest_lines(slice_idx = slice_idx, n_lines = 2)
    except NotImplementedError:
        pass

    return rois.Fan(
        largest_polygon,
        slice_idx = slice_idx,
        image = source_image,
        fan_lines=fan_lines,
        name = 'Fan-shaped body',
    )