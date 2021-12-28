from typing import Any
import holoviews as hv
import numpy as np
import colorcet
import logging

from .roi import ROI, Midline, subROI
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

    bounding_paths : hv.element.path.Path or list[tuple[hv.element.path.Path, int, int] (optional)

        Lines that define the outer edges of the fan-shaped body (to be divided angularly).
        If not a Path element itself, expected to be a tuple, where
        the first element is the holoviews Path element, the second is a slice index, and the
        third is an unimportant roi_idx that is ignored for this 

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
            bounding_paths : list[hv.element.path.Path] = None,
            **kwargs
        ):

        if not isinstance(polygon, hv.element.path.Polygons):
            raise ValueError("Fan ROI must be initialized with a polygon")
        super().__init__(polygon, **kwargs)
        self.slice_idx = slice_idx
        self.plotting_opts = {}

        if not bounding_paths is None:
            if not all(
                map(
                    lambda x: ((type(x) is hv.element.path.Path) or type(x[0]) is hv.element.path.Path),
                    bounding_paths
                )
            ):
                raise ValueError("At least one provided bounding path is not a holoviews path or a tuple containing a path as its first element.")
            self.bounding_paths = bounding_paths

        if not self.image is None:
            x_ratio = np.ptp(polygon.data[0]['x'])/self.image.shape[1] # ptp is peak-to-peak
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

    def segment(self, n_segments : int, method : str = 'triangles', viewed_from : str = 'anterior')->None:
        """
        Divides the fan in to n_segments of 'equal width', 
        defined according to the segmentation method.

        n_segments : int

            Number of columns to divide the Fan into.

        method : str (optional, default is 'triangles')

            Which method to use for segmentation. Available options:

                - triangles :

                    Requires at least two lines in the 'bounding_paths' attribute.
                    Uses the two with the largest angular span that is still less than
                    180 degrees to define the breadth of the fan. Then divides space outward
                    in evenly-spaced angular rays from the point at which the two lines intersect.
                    Looks like:  _\|/_ Columns are defined as the space between each line.

                - midline :

                    Not yet implemented, but constructs a midline through the Fan object, and divides
                    its length into chunks of equal path length. Each pixel in the Fan is assigned to
                    its nearest chunk of the midline.

        viewed_from : str (optional)

            Whether we're viewing from the anterior perspective (roi indexing should rotate counterclockwise)
            or posterior perspective (roi indixing should rotate clockwise) to match standard lab perspective.

            Options:
                
                'anterior'
                'posterior'

        Stores segments as .columns, which are a subROI class
        TODO: implement
        """
        if not type(method) is str:
            raise ValueError(f"Keyword argument method must be of type string, not type {type(method)}")

        if method == 'triangles':
            if not hasattr(self, 'bounding_paths'):
                raise NotImplementedError("Fan must have bounding paths to use triangle method of segmenting into columns.")
            self.columns = self._fit_triangles()
            

        if method == 'midline':
            raise NotImplementedError("Haven't implemented the midline method of segmenting into columns.")
            

        raise ValueError(f"Keyword argument {method} provided for method is not a valid method name.")

        
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

    def _fit_triangles(self):
        """
        Private method to fit triangles to the Fan.
        """


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

    class TriangleColumn(subROI):
        """
        Local class for Fan ROI. Defines a type of subROI in which
        the Fan is divided into triangles of equal angular width
        through the Fan. Generated by the segmentation method 'triangles'.

        Unique attributes
        -----------------

        bounding_paths : tuple[hv.element.path.Path]

            Paths outlining each triangle column.

        bounding_angles : tuple[float, float]

            Angular coordinates between the two outline rays
            defining the triangle.
        """
        def __init__(self,
                bounding_paths : tuple[hv.element.path.Path],
                bounding_angles : tuple[float],
                fan_lines : tuple[hv.element.path.Path],
                **kwargs
            ):
            super().__init__(self, **kwargs)

            self.bounding_paths = bounding_paths
            self.bounding_angles = bounding_angles

            sector_range = np.linspace(bounding_angles[0], bounding_angles[1], 60)
            
            # Define the wedge polygon
            #self.polygon = hv.Polygons(
            #    {
            #        'x' : bounding_paths[0].data[0]['x'].tolist() +
            #            [
            #                ellipse.x + (ellipse.width/2)*np.cos(offset)*np.cos(point) - (ellipse.height/2)*np.sin(offset)*np.sin(point)
            #                for point in sector_range
            #            ] +
            #            list(reversed(bounding_paths[-1].data[0]['x'])),
            #
            #        'y' : bounding_paths[0].data[0]['y'].tolist() +
            #            [
            #                ellipse.y + (ellipse.width/2)*np.sin(offset)*np.cos(point) + (ellipse.height/2)*np.cos(offset)*np.sin(point)
            #                for point in sector_range
            #            ] +
            #            list(reversed(bounding_paths[-1].data[0]['y']))
            #    }
            #)

        def visualize(self):
                return self.polygon.opts(**self.plotting_opts)

        def __repr__(self):
            """
            An triangle-defined column of the fan-shaped body
            """
            ret_str = "ROI of class TriangleColumn of a Fan\n\n"
            ret_str += f"\tCentered at {self.center()}\n"
            ret_str += f"\tOccupies angles in range {self.bounding_angles}\n"
            ret_str += f"Custom plotting options: {self.plotting_opts}\n"

            return ret_str


    class FanMidline(Midline):
        """
        Computes a midline along the long axis of the fan ROI
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args,**kwargs)
            raise NotImplementedError()
