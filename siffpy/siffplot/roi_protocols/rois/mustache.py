from typing import Any
import holoviews as hv
import numpy as np
import colorcet
import logging

from .roi import ROI, Midline, subROI
from ..extern.pairwise import pairwise

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
        super().__init__(polygon, **kwargs)
        self.slice_idx = slice_idx
        self.plotting_opts = {}
        raise NotImplementedError()
