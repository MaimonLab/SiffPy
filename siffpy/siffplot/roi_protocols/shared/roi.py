import numpy as np
import holoviews as hv

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
    def __init__(self, polygon : hv.element.path.Polygons = None):
        self.polygon = polygon

    def center(self)->tuple[float,float]:
        return (
            np.mean(self.polygon.data[0]['x']),
            np.mean(self.polygon.data[0]['y']),
        )

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
    
    segment

    """
    def __init__(
            self,
            polygon : hv.element.path.Ellipse,
            source_polygon : hv.element.path.Polygons = None,
            center_poly : hv.element.path.Polygons = None, 
            slice_idx : int = None
        ):
        if not isinstance(polygon, hv.element.path.Ellipse):
            raise ValueError("Ellipse ROI must be initialized with an Ellipse polygon")
        super().__init__(polygon)
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

    def segment(self, n_segments : int):
        """ Returns a list of ROIs corresponding to segments """
        pass
