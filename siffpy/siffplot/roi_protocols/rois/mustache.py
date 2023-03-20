from typing import Any
import holoviews as hv
import numpy as np

from siffpy.siffplot.roi_protocols.rois.roi import ROI, subROI
from siffpy.siffplot.roi_protocols.extern.pairwise import pairwise

class GlobularMustache(ROI):
    """
    A mustache-shaped ROI for individually circuled glomeruli
    """

    def __init__(
        self,
        polygon: hv.element.path.Polygons = None,
        image: np.ndarray = None,
        name: str = None,
        slice_idx: int = None,
        globular_glomeruli: list[hv.element.path.Polygons] = None,
    ):
        super().__init__(polygon, image, name, slice_idx)
        self.glomeruli = [
            GlobularMustache.GlomerulusROI(
                polygon=glom, image=image, name=name, slice_idx=slice_idx
            ) for glom in globular_glomeruli
        ]
    
    def segment(self) -> None:
        """
        Does nothing : this class is initialized with the glomeruli!
        """
        pass

    @property
    def _subROIs(self):
        return self.glomeruli

    class GlomerulusROI(subROI):
        """
        A single glomerulus
        """
        def __init__(
            self,
            polygon: hv.element.path.Polygons = None,
            image: np.ndarray = None,
            name: str = None,
            slice_idx: int = None,
            pseudophase : float = None,
        ):
            super().__init__(polygon, image, name, slice_idx)
            self.pseudophase = pseudophase


class Mustache(ROI):
    """
    Mustache-shaped ROI used for the protocerebral bridge

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

        if not isinstance(polygon, hv.element.path.Polygons):
            raise ValueError("Mustache ROI must be initialized with a polygon")
        super().__init__(polygon, slice_idx = slice_idx, **kwargs)
        self.plotting_opts = {}
        raise NotImplementedError()

    def segment(self) -> None:
        """
        TODO: come up with way to implement these
        """
        return

    class Glomerulus(subROI):
        """
        subROI class for a protocerebral bridge Mustache shape.

        TODO: IMPLEMENT
        """
        
