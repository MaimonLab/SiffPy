from typing import Any
import holoviews as hv
import numpy as np
import colorcet
import logging

from .roi import ROI, Midline, subROI
from ..extern.pairwise import pairwise
from ..utils import *

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
        VIEW_ANGLES = ['anterior', 'posterior']

        if not type(method) is str:
            raise ValueError(f"Keyword argument method must be of type string, not type {type(method)}")

        if not viewed_from in VIEW_ANGLES:
            raise ValueError(f"Keyword argument viewed_from must be in list {VIEW_ANGLES}")

        if method == 'triangles':
            if not hasattr(self, 'bounding_paths'):
                raise NotImplementedError("Fan must have bounding paths to use triangle method of segmenting into columns.")
            self._fit_triangles(n_segments, viewed_from)
            return
            
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

    def _fit_triangles(self, n_segments, viewed_from)-> list['Fan.TriangleColumn']:
        """
        Private method to fit triangle subROIs to the Fan.

        n_segments : int

            Number of segments produced in the end

        viewed_from : str

            Determines whether the left of the image
            is the fly's left or right. The subROIs are
            ordered from fly's left to fly's right, anatomically
            so this actually does matter.

            Options:

                - 'anterior'
                - 'posterior'
        """

        ## Outline
        #
        # 1) Take two lines, extend them to their intersection
        #
        # 2) Compute the angle between the lines and divide it
        # into n_segments number of sub-angles
        #
        # 3) Figure out of those angles should go from the left
        # of the image to the right of the image, or vice versa,
        # based on which way is anatomical left
        #
        # 4) Compute the directions of each ray emanating from the
        # point of intersection, and use those to compute triangles
        # intersect the Fan's polygon

        ### First find the point of intersection

        # start by extracting the bounding paths as hv.Paths
        paths = [path if type(path) == hv.Path else path[0] for path in self.bounding_paths]

        intersection = intersection_of_two_lines(*paths) # returns (x, y), not (y, x)!

        ### Find the angle swept out by the paths (using the dot product)        
        
        # find vector pointing from intersect to each path
        vectors = [vector_pointing_along_path(path, intersection) for path in paths]
        
        # The left edge is determined by whether you're facing anterior or posterior
        if viewed_from == 'anterior':
            # The left edge is closest to the x axis (arctan is close to 0)
            left_edge = vectors[np.argmin([np.abs(np.arctan2(*vector[::-1])) for vector in vectors])]
        if viewed_from == 'posterior':
            # The left edge is closest to the negative x-axis (arctan is close to +/- pi)
            left_edge = vectors[np.argmax([np.abs(np.arctan2(*vector[::-1])) for vector in vectors])]

        # Take the dot product of the two, divide by the magnitude of the vectors
        # that gives cos(angle)
        swept_angle = angle_between(*vectors)
        
        # from the anterior, it rotates with NEGATIVE theta
        if viewed_from == 'anterior':
            swept_angle *= -1.0

        rotation_angles = np.linspace(0,swept_angle, n_segments+1) # the bounds for each segment

        # vectors oriented in the direction of each bounding path
        bounding_directions = [np.dot(rotation_matrix(angle), left_edge) for angle in rotation_angles]
        paired_bounds = pairwise(bounding_directions) # pair them up, (vec1, vec2), (vec2, vec3), ...
        paired_bounding_angles = pairwise(np.linspace(0, 360, n_segments+1)) # these are nominal, span 0 to 360

        self.columns = [
            Fan.TriangleColumn(self, bound_vec, bound_ang, intersection)
            for (bound_vec, bound_ang) in zip(paired_bounds,paired_bounding_angles)
        ]

        colorwheel = colorcet.colorwheel

        idx = 0
        for column in self.columns:
            column.plotting_opts['fill_color'] = colorwheel[idx * int(len(colorwheel)/len(self.columns))]
            idx += 1

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
        """

        def __init__(self,
                fan : 'Fan',
                bounding_vectors : tuple[np.ndarray],
                bounding_angles : tuple[float, float],
                intersection_point : tuple[float, float],
                **kwargs
            ):
            """
            Initialized using the host Fan ROI, a pair of bounding rays
            projecting from the shared intersect point, nominal bounding
            angles that map from 0 to 360 across all the columns,
            and the intersection point itself

            Accepts all kwargs of the subROI class.
            """

            super().__init__(self, **kwargs)

            self.bounding_vectors = bounding_vectors
            self.bounding_angles = bounding_angles
            self.intersection_point = intersection_point

            self.polygon = polygon_bounded_by_rays(fan.polygon, self.bounding_vectors, self.intersection_point)

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