import abc, pickle, logging, os
import numpy as np
import holoviews as hv
from matplotlib.path import Path as mplPath

def apply_image(func):
    """ If the ROI has an image attribute, applies the image """
    def local_image(*args, **kwargs):
        if hasattr(args[0],'image'):
            if isinstance(args[0].image, hv.Image):
                return args[0].image * func(*args, **kwargs)
            if isinstance(args[0].image, np.ndarray):
                imshape = args[0].image.shape
                image = hv.Image(
                    {
                        'x' : np.arange(imshape[1]),
                        'y' : np.arange(imshape[0]),
                        'Intensity' : args[0].image
                    },
                    vdims=['Intensity']
                )
                if hasattr(args[0], 'plotting_opts'):
                    image = image.opts(**args[0].plotting_opts)
                return image * func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return local_image

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

    image : np.ndarray (optional)

        A template image used for masking (really anything with the right dims).

    name : str (optional)

        A name used for titling plots or for saving the roi

    plotting_opts : dict

        A dictionary that is unpacked like a holoviews opts structure when plotting.

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
    def __init__(self, polygon : hv.element.path.Polygons = None, image : np.ndarray = None, name : str = None):
        """
        Defines an ROI whose geometry is determined by the source polygon. Can be associated with
        an image to plot on top of, and can also be associated with a 'name' for when it's plotted
        and/or saved.
        """
        self.polygon = polygon
        
        if not image is None:
            # Leave it undefined otherwise -- I want errors thrown on image methods if there is no image
            self.image = image

        if not name is None:
            self.name = name
        
        self.plotting_opts = {} # called during visualize

    def center(self)->tuple[float,float]:
        """ Midpoint of the source polygon, or None if the polygon is None. """
        if self.polygon is None:
            return None
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

        if isinstance(self.polygon,hv.element.Polygons):
            poly_as_path = mplPath(list(zip(self.polygon.data[0]['x'],self.polygon.data[0]['y'])), closed=True)
        else:
            poly_as_path = mplPath(self.polygon.data[0], closed = True) # these are usually stored as arrays
       
        xx, yy = np.meshgrid(*[np.arange(0,dimlen,1) for dimlen in image.T.shape])
        x, y = xx.flatten(), yy.flatten()

        rasterpoints = np.vstack((x,y)).T

        grid = poly_as_path.contains_points(rasterpoints)
        grid = grid.reshape(image.shape)
        
        return grid

    def get_subroi_masks(self, image : np.ndarray = None, ret_type : type = list) -> list[np.ndarray]:
        """
        Returns a list or array (depending on keyword argument ret_type) of the numpy masks of
        all subROIs of this ROI. If the ROI does not have an assigned 'image' attribute, it can
        also be provided as a keyword argument with keyword image.

        Arguments
        ---------

        image : np.ndarray

            A template image that provides the dimensions of the image that the mask needs to be
            embedded in

        ret_type : type

            Can be any of
                - list
                - numpy.ndarray
                - 'list'
                - 'array'

        """
        if not hasattr(self,'subROIs'):
            raise AttributeError("ROI does not have any assigned subROIs")

        if image is None and not hasattr(self,'image'):
            raise ValueError("No template image provided!")
        if image is None:
            image = self.image

        if ret_type == list or ret_type == 'list':
            return [subroi.mask(image=image) for subroi in self.subROIs]
        if ret_type == np.ndarray or ret_type == 'array':
            return np.array([subroi.mask(image=image) for subroi in self.subROIs])
        raise ValueError(f"Argument rettype is {ret_type}. rettype must be either list or np.ndarray")
    
    @apply_image
    def visualize(self, **kwargs) -> hv.Element:
        """
        Returns a holoviews element to compose for visualization
        """
        # add your own polygon
        poly = self.polygon.opts(**self.plotting_opts)
        if hasattr(self, 'subROIs'):
            
            SUBOPTS = {
                'line_color' : 'white',
                'line_dash'  : 'dashed',
                'fill_alpha' : 0.3
            }
            
            if 'subopts' in kwargs:
                # customizable
                for key, value in kwargs['subopts']:
                    SUBOPTS[key] = value

            poly *= self.subROIs[0].visualize().opts(**SUBOPTS)
            for polyidx in range(1,len(self.subROIs)):
                poly *= self.subROIs[polyidx].visualize().opts(**SUBOPTS)

        # if not, just return the outline.
        return poly,opts(**self.plotting_opts)

    def find_midline(self):
        """
        The most primitive midline for an ROI takes two
        previously selected points and draws a line to connect
        them. This is the default midline, and what is returned
        for any ROI which does not define a midline function itself.

        If there are no selected points, I intend to make this default
        to the 'find the points farthest from the boundary which best
        span the structure' method, but I haven't implemented that yet.
        """
        logging.warn("""\n\n
            WARNING:\n\tThis is the SUPERCLASS ROI find_midline method.\n
            \tThis means whatever ROI class you're using has not overwritten\n
            \tthe simple linear fit between two endpoints method. Be aware!\n\n
            """)

        if not hasattr(self,'selected_points'):
            raise AttributeError("No points selected for drawing a midline.")
        
        if len(self.selected_points) < 2:
            raise AttributeError("Need at least two selected points to draw a midline.")

        if len(self.selected_points) > 2:
            logging.warn("More than two selected points. Using only the last two.")

        endpts = (self.selected_points[-2], self.selected_points[-1])
        
        def straight_line(endpts, t):
            T = t[-1] - t[0]
            return (
                endpts[0][0] + (t/T)*(endpts[1][0]-endpts[0][0]),
                endpts[0][1] + (t/T)*(endpts[1][1]-endpts[0][1])
            ) 
        
        self.midline = Midline(self, fmap = lambda t: straight_line(endpts,t))
        return self.midline

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
        if hasattr(self,'name'):
            file_name += str(self.name)
        else:
            file_name += str(self.polygon.__hash__())
        with open(file_name + ".roi",'wb') as roi_file:
            pickle.dump(self, roi_file)

    def __repr__(self)->str:
        """
        Pretty summary of an ROI
        """
        return f"ROI superclass"

    def __getitem__(self, key):
        """
        This is for when ROIs are treated
        like lists even if they aren't one.
        This is to make all code able to handle
        either the case where it assumes there's
        only one ROI referenced by a SiffPlotter
        or the case where it assumes
        there are several.
        """
        if key == 0:
            return self
        else:
            raise TypeError("'ROI' object is not subscriptable (except with 0 to return itself)")

    def __iter__(self) :
        return iter([self])

class subROI(ROI):
    """
    A subclass of the ROI designed solely to indicate
    to analysis functions that this type of ROI is a segment
    or cluster of a larger ROI, e.g. for HeatMap type
    plotting outputs.

    So far, no custom functionality other than it being
    a subclass identifiable with isinstance.
    """
    def __init__(self, *args, **kwargs):
        """ Standard ROI init """
        ROI.__init__(self, *args, **kwargs)



class Midline():
    """
    Midlines are a common structure I'm finding myself using.
    Thought it would make sense to turn it into a class.

    Attributes
    ----------
    t : np.ndarray

        Parameterization of the midline, numpy array running from 0 to 2*np.pi

    fmap : function

        Takes self.t to the midline structure. Should return a list or tuple
        of length two, each containing 'x's or 'y's, one for every value of t.
        So it should accept a numpy array

    Methods
    -------
    fit() : 

        Not yet implemented

    draw(no_overlay = False) :

        Returns a hv.Path that traces the midline. If no_overlay is True, then returns
        just the path. Otherwise returns an hv.Overlay
    """
    def __init__(self, source_roi : ROI, point_count : int = 360, fmap = None):
        self.source_roi = source_roi
        self.t = np.linspace(0,2*np.pi, point_count)
        self.fmap = fmap

    def fit(self, cost) -> None:
        raise NotImplementedError()

    def draw(self, no_overlay = False)->hv.element.path.Path:
        pts = self.fmap(self.t)
        if no_overlay:
            return hv.Path(
            (pts[0],
            pts[1],
            self.t)
        )
        try:
            return self.source_roi.visualize() * hv.Path(
                (pts[0],
                pts[1],
                self.t)
            )
        except:
            # Perhaps visualize is not defined for this ROI class
            return self.source_roi.visualize() * hv.Path(
                (pts[0],
                pts[1],
                self.t)
            )

    @abc.abstractmethod
    def mask(self) -> None:
        """
        Haven't decided, should this have a definition in the base class?
        Or should I make this an abstract method?
        """
        raise NotImplementedError()