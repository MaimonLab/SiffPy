from typing import Any
import holoviews as hv
import numpy as np
from scipy.stats import circmean
import colorcet

from .roi import ROI, Midline, subROI, apply_image, ViewDirection
from ..extern.pairwise import pairwise

EB_OFFSET = (1/2) * np.pi # EPG going to left of each PB (when viewed from posterior) is the first ROI

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
        super().__init__(polygon, slice_idx = slice_idx, **kwargs)
        self.source_polygon = source_polygon
        self.center_poly = center_poly
        self.plotting_opts = {}

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

        Returns
        -------

        None

        """
        self.midline = self.RingMidline(self)

    def segment(self, n_segments : int = 16, viewed_from : ViewDirection = ViewDirection.ANTERIOR)->None:
        """
        Creates an attribute wedges, a list of WedgeROIs corresponding to segments
        
        PARAMETERS
        ----------

        n_segments : int
        
            Number of wedges to produce

        viewed_from : str (optional)

            Whether we're viewing from the anterior perspective (roi indexing should rotate counterclockwise)
            or posterior perspective (roi indixing should rotate clockwise) to match standard lab perspective.

            Options:
                
                'anterior'
                'posterior'
        
        """

        cx, cy = self.center()
        ell = self.polygon

        if (viewed_from == ViewDirection.ANTERIOR) or (viewed_from == ViewDirection.ANTERIOR.value):
            angles = np.linspace(EB_OFFSET, EB_OFFSET - 2*np.pi, n_segments+1)
        elif (viewed_from == ViewDirection.POSTERIOR) or (viewed_from == ViewDirection.POSTERIOR.value):
            angles = np.linspace(EB_OFFSET , EB_OFFSET + 2*np.pi, n_segments+1)
        else:
            raise ValueError(f"Argument 'viewed_from' is {viewed_from}, must be in {[x.value for x in ViewDirection]}")
            
        self.perspective = viewed_from
        offset = ell.orientation

        # Go 360/n_segments degrees around the ellipse
        # And draw a dividing line at the end
        # The WedgeROI will build an ROI out of the sector
        # of the ellipse between its input line boundaries
        dividing_lines = [
            hv.Path(
                {
                    'x':[cx, ell.x + (ell.width/2)*np.cos(offset)*np.cos(angle) - (ell.height/2)*np.sin(offset)*np.sin(angle)] ,
                    'y':[cy, ell.y + (ell.width/2)*np.sin(offset)*np.cos(angle) + (ell.height/2)*np.cos(offset)*np.sin(angle)]
                }
            )
            for angle in angles
        ]

        image = None
        if hasattr(self,'image'):
            image = self.image
        self.wedges = [
            Ellipse.WedgeROI(
                boundaries[0],
                boundaries[1],
                ell,
                image=image,
                slice_idx = self.slice_idx
            )
            for boundaries in zip(tuple(pairwise(dividing_lines)),tuple(pairwise(angles)))
        ]

        colorwheel = colorcet.colorwheel

        idx = 0
        for wedge in self.wedges:
            wedge.plotting_opts['fill_color'] = colorwheel[idx * int(len(colorwheel)/len(self.wedges))]
            wedge.plotting_opts['fill_alpha'] = 0.3
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

    def mask(self, image : np.ndarray = None)->np.ndarray:
        """Uses the default mask but then subtracts out center polygon if it has one"""
        grid = super().mask()
        if self.center_poly is None:
            return grid
        if image is None and hasattr(self,'image'):
            image = self.image

        from matplotlib.path import Path as mplPath
        
        if isinstance(self.center_poly,hv.element.Polygons):
            poly_as_path = mplPath(list(zip(self.center_poly.data[0]['x'],self.center_poly.data[0]['y'])), closed=True)
        else:
            poly_as_path = mplPath(self.center_poly.data[0], closed = True) # these are usually stored as arrays
       
        xx, yy = np.meshgrid(*[np.arange(0,dimlen,1) for dimlen in image.T.shape])
        x, y = xx.flatten(), yy.flatten()

        rasterpoints = np.vstack((x,y)).T

        inner_grid = poly_as_path.contains_points(rasterpoints)
        inner_grid = inner_grid.reshape(image.shape)
        grid[inner_grid] = False

        return grid

    @apply_image
    def visualize(self, **kwargs)->hv.Element:
        """
        Kwargs options:

        wedgeopts : dict

            Dictionary of arguments passed to the .opts call for each wedge's polygon

        """
        if hasattr(self, 'wedges'):
            
            WEDGEOPTS = {
                'line_color' : 'white',
                'line_dash'  : 'dashed',
                'fill_alpha' : 0.3
            }
            
            if 'wedgeopts' in kwargs:
                # customizable
                for key, value in kwargs['wedgeopts']:
                    WEDGEOPTS[key] = value

            poly = self.wedges[0].visualize().opts(**WEDGEOPTS)
            for polyidx in range(1,len(self.wedges)):
                poly *= self.wedges[polyidx].visualize().opts(**WEDGEOPTS)

            # now add your own polygon
            poly *= self.polygon.opts(**self.plotting_opts)
            return poly
        
        # if not, just return the outline.
        return self.polygon.opts(**self.plotting_opts)

    def __getattr__(self, attr)->Any:
        if attr == '_subROIs':
            if hasattr(self,'wedges'):
                return self.wedges
        else:
            return object.__getattribute__(self, attr)

    def __repr__(self)->str:
        """
        A few summary values
        """
        ret_str = "ROI of class Ellipse\n\n"
        ret_str += f"\tCentered at {self.center()}\n"
        ret_str += f"\tRestricted to slice(s) {self.slice_idx}\n"
        if hasattr(self, 'wedges'):
            ret_str += f"\tSegmented into {len(self.wedges)} wedges\n"
        if hasattr(self,'perspective'):
            ret_str += f"\tViewed from {self.perspective} direction\n"
        if hasattr(self,'midline'):
            ret_str += f"Midline defined as\n"
        ret_str += f"Custom plotting options: {self.plotting_opts}\n"

        return ret_str

    class WedgeROI(subROI):
        """
        Local class for ellipsoid body wedges. Very simple

        Takes two lines and an ellipse, with the lines defining
        the edges of the sector the WedgeROI occupies. Then it
        returns a subROI whose polygon is approximately the interior
        of the Ellipse in between the two dividing line segments.

        Unique attributes
        -----------------

        bounding_paths : tuple[hv.element.Path] 

            The edges of the wedge that divide the
            ellipse into segments

        bounding_angles : tuple[float]

            The angular value along the outer contour
            of the ellipse that correspond to the edge
            bounding_paths
        """
        def __init__(self,
                bounding_paths : tuple[hv.element.Path],
                bounding_angles : tuple[float],
                ellipse : hv.element.path.Ellipse,
                slice_idx : int = None,
                **kwargs
            ):
            super().__init__(self, **kwargs)

            self.bounding_paths = bounding_paths
            self.bounding_angles = bounding_angles
            self.slice_idx = slice_idx

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
        
        def visualize(self):
            return self.polygon.opts(**self.plotting_opts)

        @property
        def angle(self):
            return circmean(self.bounding_angles)

        def __repr__(self):
            """
            An ellipse's wedge.
            """
            ret_str = "ROI of class Wedge of an Ellipse\n\n"
            ret_str += f"\tCentered at {self.center()}\n"
            ret_str += f"\tOccupies angles in range {self.bounding_angles}\n"
            ret_str += f"Custom plotting options: {self.plotting_opts}\n"

            return ret_str


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
        
        def draw(self):
            return self.path

        def mask(self)->None:
            """
            TODO
            """
            raise NotImplementedError()