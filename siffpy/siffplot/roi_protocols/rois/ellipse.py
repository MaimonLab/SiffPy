from typing import Any
import holoviews as hv
import numpy as np
import colorcet

from .roi import ROI, Midline, subROI
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
        super().__init__(polygon, **kwargs)
        self.source_polygon = source_polygon
        self.center_poly = center_poly
        self.slice_idx = slice_idx
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
        """
        self.midline = self.RingMidline(self)

    def segment(self, n_segments : int, viewed_from = 'anterior')->None:
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

        if viewed_from == 'anterior':
            angles = np.linspace(EB_OFFSET, EB_OFFSET + 2*np.pi, n_segments+1)
        elif viewed_from == 'posterior':
            angles = np.linspace(EB_OFFSET , EB_OFFSET - 2*np.pi, n_segments+1)
        else:
            raise ValueError(f"Argument 'viewed_from' is {viewed_from}, must be 'anterior' or 'posterior'")
            
        self.perspective = viewed_from
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

        image = None
        if hasattr(self,'image'):
            image = self.image
        self.wedges = [
            self.WedgeROI(
                boundaries[0],
                boundaries[1],
                ell,
                image=image
            )
            for boundaries in zip(tuple(pairwise(dividing_lines)),tuple(pairwise(angles)))
        ]

        colorwheel = colorcet.colorwheel

        idx = 0
        for wedge in self.wedges:
            wedge.plotting_opts['fill_color'] = colorwheel[idx * int(len(colorwheel)/len(self.wedges))]
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
        if attr is 'subROIs':
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
        
        def visualize(self):
            return self.polygon.opts(**self.plotting_opts)

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

        def get_masks(self)->None:
            """
            TODO
            """
            pass