# Shared code for ROI extraction

# To future implementers: OPTS should only be applied
# in the function visualize, which can only be called if
# holoviews has initialized its extension. This maintains
# full compatibility with napari and other %gui qt frameworks
# in Jupyter. The local_opts arguments can be converted into
# napari properties and so this framework is general to both.

import numpy as np
import matplotlib

from siffpy.siffplot.roi_protocols.extern.smallest_circle import make_circle
from siffpy.siffplot.roi_protocols.rois.roi import *
from siffpy.siffplot.roi_protocols.rois.ellipse import *
from siffpy.siffplot.roi_protocols.rois.fan import *
from siffpy.siffplot.roi_protocols.rois.blob import *
from siffpy.siffplot.roi_protocols.rois.mustache import *


__all__ = [
    'polygon_area',
    'fit_ellipse_to_poly',
    'get_largest_polygon',
]

def contains_points(polygon, points)->np.ndarray:
    """
    Returns an array of bools reflecting whether
    each of an array of points is contained in a polygon
    """

    mplPath = matplotlib.path.Path(
        np.array(
            [
                polygon.data[0]['x'],
                polygon.data[0]['y']
            ]
        ).T
    )

    # in case it's an iterable but not array
    points = np.array(points)
    if len(points.shape) == 1:
        points = [points] # need to be 2d!

    return mplPath.contains_points(points)

def point_to_line_distance(point, endpoints):
    """
    Endpoints is a pair of points
    Uses formula at https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    """
    return np.abs(
        (endpoints[1][0] - endpoints[0][0]) * (endpoints[0][1] - point[1]) - # (x2-x1)(y1-y0)
        (endpoints[0][0] - point[0]) * (endpoints[1][1] - endpoints[0][1]) # (x1-x0)(y2-y1)
    )/np.sqrt(
        (endpoints[1][0]-endpoints[0][0])**2 +
        (endpoints[1][1]-endpoints[0][1])**2
    )

def point_to_polygon_distance(point, polygon):
    """
    Iterates point_to_line_distance across all pairs of edges in a polygon.
    Returns the smallest one.
    """
    to_points = list(zip(polygon.data[0]['x'],polygon.data[0]['y']))
    segment_endpts = list(pairwise(to_points)) + [(to_points[-1],to_points[0])]

    return np.nanmin([point_to_line_distance(point,ept) for ept in segment_endpts])

def polygon_area(x_coords : np.ndarray, y_coords : np.ndarray) -> float:
    """ Shoelace method to compute area"""
    return 0.5*np.abs(
        np.sum(x_coords*np.roll(y_coords,1)-y_coords*np.roll(x_coords,1))
    )
    
def annotation_dict_to_numpy(annotation_dict : dict, slice_idx : int) -> np.ndarray:
    """ Returns the numpy array underlying the image in a single frame of an annotation dict"""
    return annotation_dict[slice_idx]['layout']['DynamicMap'].I.data[()].Image.I.data['Intensity'] # yeah, I know...