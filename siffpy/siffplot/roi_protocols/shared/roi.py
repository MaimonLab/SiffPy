import itertools

import numpy as np
import holoviews as hv
from matplotlib.path import Path as mplPath

__all__ = [
    'Ellipse'
]

class ROI():
    """
    Class for an ROI. Contains information about bounding box, brain region to which
    this ROI belongs, method produced to extract this ROI, and probably information about
    how to use it for computations. May be extended by future use cases.

    .........

    Attributes
    ----------

    polygon : hv.element.path.Polygons

        A polygon representing the bounds of the ROI


    """
    def __init__(self, polygon : hv.element.path.Polygons = None, image : np.ndarray = None):
        self.polygon = polygon
        
        if not image is None:
            # Leave it undefined otherwise -- I want errors thrown on image methods if there is no image
            self.image = image

    def center(self)->tuple[float,float]:
        return (
            np.mean(self.polygon.data[0]['x']),
            np.mean(self.polygon.data[0]['y']),
        )

    def mask(self, image : np.ndarray = None)->np.ndarray:
        """
        Returns a mask of the polygon, True inside and False outside.
        Needs an image to define the bounds, if one hasn't been provided to the ROI before
        """
        if not hasattr(self, 'image') and image is None:
            raise ValueError("No base image provided to ROI for masking")

        if image is None and hasattr(self,'image'):
            image = self.image

        poly_as_path = mplPath(list(zip(self.polygon.data[0]['x'],self.polygon.data[0]['y'])), closed=True)
       
        xx, yy = np.meshgrid(*[np.arange(0,dimlen,1) for dimlen in image.shape])
        x, y = xx.flatten(), yy.flatten()

        rasterpoints = np.vstack((x,y)).T

        grid = poly_as_path.contains_points(rasterpoints)
        grid = grid.reshape(image.shape)
        
        return grid

    def opts(self, *args, **kwargs)->None:
        """ Wraps the self.polygon's opts """
        self.polygon.opts(*args, **kwargs)

class Ellipse(ROI):
    """
    Ellipse-shaped ROI

    ........

    Attributes
    ----------

    polygon        : hv.element.path.Ellipse

        A HoloViews Ellipse representing the region of interest.

    source_polygon : hv.element.path.Polygons

        The polygon used to originally create the Ellipse

    center_poly    : hv.element.path.Polygons

        An optional polygon that demarcates where the hole in the ellipse should be.

    slice_idx      : int

        Integer reference to the z-slice that the source polygon was drawn on.

    .......

    Methods
    ------------

    compute midline() -> None

        Creates self attribute midline that is designed to be
        between the center of the ellipse and its outline

    segment

    """
    def __init__(
            self,
            polygon : hv.element.path.Ellipse,
            source_polygon : hv.element.path.Polygons = None,
            center_poly : hv.element.path.Polygons = None, 
            slice_idx : int = None,
            **kwargs
        ):
        if not isinstance(polygon, hv.element.path.Ellipse):
            raise ValueError("Ellipse ROI must be initialized with an Ellipse polygon")
        super().__init__(polygon, **kwargs)
        self.source_polygon = source_polygon
        self.center_poly = center_poly
        self.slice_idx = slice_idx

    def center(self)->tuple[float, float]:
        """ Returns a tuple of the x and y coordinates of the Ellipse center """
        if self.center_poly is None:
            return (self.polygon.x, self.polygon.y)
        else:
            # uses the mean of the vertices, rather than the center of the smallest circle.
            verts = self.center_poly.data[0]
            return (np.mean(verts['x']),np.mean(verts['y']))

    def compute_midline(self)->None:
        """
        Computes the midline of the Ellipse, stores it in the attribute midline
        """
        raise NotImplementedError()

    def segment(self, n_segments : int)->None:
        """ Creates an attribute wedges, a list of WedgeROIs corresponding to segments """

        cx, cy = self.center()
        ell = self.polygon

        angles = np.linspace(0, 2*np.pi, n_segments+1)
        offset = ell.orientation

        dividing_lines = [
            hv.Path(
                {
                    'x':[cx, ell.x + (ell.width/2)*np.cos(offset)*np.cos(angle) - (ell.height/2)*np.sin(offset)*np.sin(angle)] ,
                    'y':[cy, ell.y + (ell.width/2)*np.sin(offset)*np.cos(angle) + (ell.height/2)*np.cos(offset)*np.sin(angle)]
                }
            )
            for angle in angles
        ]

        self.wedges = [
            self.WedgeROI(
                boundaries[0],
                boundaries[1],
                ell
            )
            for boundaries in zip(tuple(pairwise(dividing_lines)),tuple(pairwise(angles)))
        ]

    def get_roi_masks(self, n_segments : int = 16, image : np.ndarray = None)->list:
        if image is None and not hasattr(self,'image'):
            raise ValueError("No template image provided!")
        if image is None:
            image = self.image

        if not hasattr(self, 'wedges'):
            self.segment(n_segments)
        return [wedge.mask(image=image) for wedge in self.wedges]

    class WedgeROI(ROI):
        """
        Local class for ellipsoid body wedges. Very simple
        """
        def __init__(self,
                bounding_paths : tuple[hv.element.Path],
                bounding_angles : tuple[float],
                ellipse : hv.element.path.Ellipse,
                **kwargs
            ):
            super().__init__(self, **kwargs)

            self.bounding_paths = bounding_paths
            self.bounding_angles = bounding_angles

            sector_range = np.linspace(bounding_angles[0], bounding_angles[1], 60)
            offset = ellipse.orientation

            # Define the wedge polygon
            self.polygon = hv.Polygons(
                {
                    'x' : bounding_paths[0].data[0]['x'].tolist() +
                        [
                            ellipse.x + (ellipse.width/2)*np.cos(offset)*np.cos(point) - (ellipse.height/2)*np.sin(offset)*np.sin(point)
                            for point in sector_range
                        ] +
                        list(reversed(bounding_paths[-1].data[0]['x'])),

                    'y' : bounding_paths[0].data[0]['y'].tolist() +
                        [
                            ellipse.y + (ellipse.width/2)*np.sin(offset)*np.cos(point) + (ellipse.height/2)*np.cos(offset)*np.sin(point)
                            for point in sector_range
                        ] +
                        list(reversed(bounding_paths[-1].data[0]['y']))
                }
            )


class Midline():
    """
    Midlines are a common structure I'm finding myself using.
    Thought it would make sense to turn it into a class.

    Attributes
    ----------
    t : np.ndarray

        Parameterization of the midline

    fmap : function

        Takes self.t to the midline structure. Probably a holoviews.elements.paths.Path in the end?
    """
    def __init__(self, point_count : int = 360, fmap = None):
        self.t = np.linspace(0,2*np.pi, point_count)
        self.fmap = fmap

# Itertools recipe
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)