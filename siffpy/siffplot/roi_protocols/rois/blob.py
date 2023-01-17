from typing import Any
import holoviews as hv
import numpy as np

from siffpy.siffplot.roi_protocols.rois.roi import ROI, subROI, ViewDirection
from siffpy.siffplot.roi_protocols.extern.pairwise import pairwise

class Blobs(ROI):
    """
    Blob-shaped ROIs used for the noduli.
    Blobs are boring, and their main feature
    is their pair of polygons. They do not have a midline!

    ........

    Attributes
    ----------

    polygon        : hv.element.path.Polygons

        A HoloViews Polygon representing the region of interest.

    slice_idx      : int

        Integer reference to the z-slice that the source polygon was drawn on.

    .......

    Methods
    ------------

    compute midline() -> None

        Not yet implemented 

    segment(n_segments) -> None

        Not yet implemented

    get_roi_masks(image) -> list[np.ndarray]

        Returns a list (or np.ndarray) of the masks for all wedge parameters (if they exist)
    """

    def __init__(
            self,
            polygon : hv.element.path.Polygons,
            slice_idx : int = None,
            **kwargs
        ):

        if not isinstance(polygon, hv.element.Polygons):
            raise ValueError("Blobs ROI must be initialized with a polygons object!")
        super().__init__(polygon, slice_idx = slice_idx, **kwargs)
        self.plotting_opts = {}

    def segment(self, viewed_from : ViewDirection = ViewDirection.ANTERIOR, **kwargs) -> None:
        """ n_segments is not a true keyword param, always produces two """
        self.hemispheres = [Blobs.Hemisphere(pgon) for pgon in self.polygon.split()]
        # TODO: USE BOUNDING ANGLES AND ORDER LEFT TO RIGHT

    def visualize(self) -> hv.Element:
        return self.polygon.opts(**self.plotting_opts)

    def find_midline(self):
        """ No midline! """
        raise ValueError("Blob type ROIs do not have a midline!")

    def __getattr__(self, attr)->Any:
        """
        Custom subROI call to return hemispheres
        as the subROI
        """
        if attr == '_subROIs':
            if hasattr(self,'hemispheres'):
                return self.hemispheres
            else:
                raise AttributeError(f"No hemispheres attribute assigned for Blobs")
        else:
            return object.__getattribute__(self, attr)

    class Hemisphere(subROI):
        """ A vanilla ROI, except it's classed as a subROI. Has just a polygon. """
        def __init__(self,
                polygon,
                **kwargs
            ):
            super().__init__(self, polygon = polygon, **kwargs)