from typing import Any
import holoviews as hv
import numpy as np
import colorcet
import logging

from .roi import ROI, Midline
from ..extern.pairwise import pairwise

class Fan(ROI):
    """
    Fan-shaped ROI

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
            raise ValueError("Fan ROI must be initialized with a polygon")
        super().__init__(polygon, **kwargs)
        self.slice_idx = slice_idx
        self.plotting_opts = {}

        if not self.image is None:
            x_ratio = np.ptp(polygon.data[0]['x'])/self.image.shape[1]
            y_ratio = np.ptp(polygon.data[0]['y'])/self.image.shape[0]
            if y_ratio > x_ratio:
                self.long_axis = 'x'
            else:
                self.long_axis = 'y'
        else:
            # Making an assumption later. Maybe I'll make things programmatically
            # look out for this?
            self.long_axis = None

    def visualize(self)->hv.Element:
        return self.polygon.opts(**self.plotting_opts)

    def segment(self, n_segments : int, viewed_from : str = 'anterior')->None:
        """
        Divides the fan in to n_segments of 'equal width', 
        defined as...

        viewed_from : str (optional)

            Whether we're viewing from the anterior perspective (roi indexing should rotate counterclockwise)
            or posterior perspective (roi indixing should rotate clockwise) to match standard lab perspective.

            Options:
                
                'anterior'
                'posterior'

        Stores segments as .columns, which are a subROI class
        TODO: implement
        """
        raise NotImplementedError()

    def find_midline(self)->None:
        """
        Returns a midline through the fan-shaped body if at least two points
        are defined on the edge of the ROI. Uses ??? method
        """
        if not hasattr(self, 'selected_points'):
            raise AttributeError(f"No points selected on ROI \n{self}")

        if len(self.selected_points) > 2:
            logging.warn(f"More than two points selected on ROI. Using last two defined. ROI:\n{self}")
        
        if len(self.selected_points) < 2:
            raise RuntimeError(f"Fewer than two points defined on ROI:\n{self}")


        raise NotImplementedError()

    def __getattr__(self, attr)->Any:
        """
        Custom subROI call to return columns
        as the subROI
        """
        if attr == 'subROIs':
            if hasattr(self,'columns'):
                return self.columns
            else:
                raise AttributeError(f"No columns attribute assigned for Fan")
        else:
            return object.__getattribute__(self, attr)

    def __repr__(self)->str:
        """
        A few summary values
        """
        ret_str = "ROI of class Fan\n\n"
        ret_str += f"\tCentered at {self.center()}\n"
        ret_str += f"\tRestricted to slice(s) {self.slice_idx}\n"
        if hasattr(self, 'columns'):
            ret_str += f"\tSegmented into {len(self.columns)} columns\n"
        if hasattr(self,'perspective'):
            ret_str += f"\tViewed from {self.perspective} direction\n"
        if hasattr(self,'midline'):
            ret_str += f"Midline defined as\n"
        ret_str += f"Custom plotting options: {self.plotting_opts}\n"

        return ret_str

    class FanMidline(Midline):
        """
        Computes a midline along the long axis of the fan ROI
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args,**kwargs)
            raise NotImplementedError()
