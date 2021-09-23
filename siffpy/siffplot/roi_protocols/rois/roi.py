from ..extern.pairwise import pairwise

import abc, pickle, logging, os
import numpy as np
import holoviews as hv
import colorcet
from matplotlib.path import Path as mplPath

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
        
        self.plotting_opts = {} # called during visualize

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
    
    @abc.abstractmethod
    def visualize(self) -> hv.Element:
        """
        Returns a holoviews element to compose for visualization
        """
        pass

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
        if not os.path.exists(path):
            os.makedirs(path)
            pass
        file_name = os.path.join(path,self.__class__.__name__)
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