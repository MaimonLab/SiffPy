from ..extern.pairwise import pairwise

import abc, pickle, logging
import numpy as np
import holoviews as hv
import colorcet
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

    image : np.ndarray

        A template image used for masking (really anything with the right dims).

    ......
    Methods
    -------

    center()->(center_x, center_y)

        Returns the mean coordinates of the polygon, if not overwritten by an inheriting class

    mask(image)->masked_array

        Returns a numpy array that is True where the array is contained by the polygon of the ROI

    opts(*args, **kwargs)

        Applys the holoviews opts supplied to the ROI's polygon


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

        # Very easy to accidentally pass in the HoloViews object instead.
        if isinstance(image, hv.Element):
            logging.warning(f"Provided image is of type {type(image)}, not a numpy array. Attempting to convert.")
            if not isinstance(image, hv.element.Raster):
                raise ValueError("Incompatible type of HoloViews object (must be a raster, not vectorized).")
            try:
                image = np.ones(
                    (
                        np.max(image.data['x']) - np.min(image.data['x']) + 1,
                        np.max(image.data['y']) - np.min(image.data['y']) + 1
                    ), dtype=np.uint8)
            except Exception as e:
                raise ValueError(f"Incompatible HoloViews object.\nException:\n\t{e}")

        poly_as_path = mplPath(list(zip(self.polygon.data[0]['x'],self.polygon.data[0]['y'])), closed=True)
       
        xx, yy = np.meshgrid(*[np.arange(0,dimlen,1) for dimlen in image.shape])
        x, y = xx.flatten(), yy.flatten()

        rasterpoints = np.vstack((x,y)).T

        grid = poly_as_path.contains_points(rasterpoints)
        grid = grid.reshape(image.shape)
        
        return grid

    def draw_midline(self):
        """ TODO """
        raise NotImplementedError()

    def opts(self, *args, **kwargs)->None:
        """ Wraps the self.polygon's opts """
        self.polygon.opts(*args, **kwargs)

    def save(self, path)->None:
        """
        Saves the ROIs as .roi files. These files are just a pickled
        version of the actual ROI object. ROI name is mangled with 
        unique attributes about the ROI so that no two will overlap
        by using the same name.
        """
        file_name = path + self.__class__.__name__ 
        file_name += str(self.polygon.__hash__())
        with open(file_name + ".roi",'wb') as roi_file:
            pickle.dump(self, roi_file)



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
    def __init__(self, source_roi : ROI, point_count : int = 360, fmap = None):
        self.source_roi = source_roi
        self.t = np.linspace(0,2*np.pi, point_count)
        self.fmap = fmap

    def fit(self, cost) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_masks(self) -> None:
        """
        Haven't decided, should this have a definition in the base class?
        Or should I make this an abstract method?
        """
        raise NotImplementedError()

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

    center()->(center_x, center_y)

        If the center polygon is defined, returns the center of that polygon, rather than the main ellipse.

    compute midline() -> None

        Creates self attribute midline that is designed to be
        between the center of the ellipse and its outline

    segment(n_segments) -> None

        Creates the attribute 'wedges' as a list of length n_segments. Each wedge is
        a WedgeROI, evenly dividing the ellipse into n_segment pieces.

    get_roi_masks(image) -> list[np.ndarray]

        Returns a list (or np.ndarray) of the masks for all wedge parameters (if they exist)

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
        self.midline = self.RingMidline(self)

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

        colorwheel = colorcet.colorwheel

        idx = 0
        for wedge in self.wedges:
            wedge.opts(fill_color = colorwheel[idx * int(len(colorwheel)/len(self.wedges))])
            idx += 1

    def get_roi_masks(self, n_segments : int = 16, image : np.ndarray = None, rettype = list)->list[np.ndarray]:
        if image is None and not hasattr(self,'image'):
            raise ValueError("No template image provided!")
        if image is None:
            image = self.image

        if not hasattr(self, 'wedges'):
            self.segment(n_segments)

        if rettype == list:
            return [wedge.mask(image=image) for wedge in self.wedges]
        if rettype == np.ndarray:
            return np.array([wedge.mask(image=image) for wedge in self.wedges])
        raise ValueError(f"Argument rettype is {rettype}. rettype must be either list or np.ndarray")

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


    class RingMidline(Midline):
        """
        A ring-shaped midline specific for the ellipsoid body.

        Simple to parameterize, so might be able to avoid all the gradient mess.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            c_x, c_y = self.source_roi.center()
            ellipse = self.source_roi.polygon
            offset = ellipse.orientation
            angles = self.t / (2*np.pi)
            self.path = hv.Path(
                { # halfway between the center and the boundary of the outer ellipse
                    'x':[
                        0.5*(c_x + ellipse.x + 
                            (
                                (ellipse.width/2)*np.cos(offset)*np.cos(angle) - 
                                (ellipse.height/2)*np.sin(offset)*np.sin(angle)
                            )
                            )
                        for angle in angles
                    ],
                    
                    'y':[
                        0.5*(c_y + ellipse.y + 
                            (
                                (ellipse.width/2)*np.sin(offset)*np.cos(angle) + 
                                (ellipse.height/2)*np.cos(offset)*np.sin(angle)
                            )
                            ) 
                        for angle in angles
                    ]
                }
            )
        
        def get_masks(self)->None:
            """
            TODO
            """
            pass


