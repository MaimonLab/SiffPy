# Shared code for ROI extraction

# To future implementers: OPTS should only be applied
# in the function visualize, which can only be called if
# holoviews has initialized its extension. This maintains
# full compatibility with napari and other %gui qt frameworks
# in Jupyter. The local_opts arguments can be converted into
# napari properties and so this framework is general to both.

import numpy as np
import matplotlib
import holoviews as hv

from siffpy.siffplot.roi_protocols.extern.smallest_circle import make_circle
from siffpy.siffplot.roi_protocols.rois.roi import *
from siffpy.siffplot.roi_protocols.rois.ellipse import *
from siffpy.siffplot.roi_protocols.rois.fan import *
from siffpy.siffplot.roi_protocols.rois.blob import *

try:
    from siffpy.siffplot.roi_protocols.utils.napari_fcns import * # only works if napari is installed
except ImportError:
    pass



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

def fit_ellipse_to_poly(poly : hv.element.path.Polygons, method : str = 'lsq', center : str = 'circle') -> hv.element.path.Ellipse:
    """
    Returns an Ellipse polygon fit to a regular polygon, using the method in the
    argument method. The center of the ellipse is defined by the string center.

    Options for method:
        - lsq : Uses least squares regression
        - SVD : Uses singular value decomposition

    Options for center:

        - circle : Uses the center of the smallest circle fitting the data
        - mean   : Uses the mean of the vertices' positions
    """

    vertices = poly.data[0]
    #Find the smallest circle enclosing all vertices -- its center is the midpoint of the ellipse.
    if center.lower() == 'circle':
        (center_x, center_y, _) = make_circle(list(zip(vertices['x'],vertices['y']))) # an extern function
    elif center.lower() == 'mean':
        (center_x, center_y) = np.mean(vertices['x']), np.mean(vertices['y'])

    if method == 'SVD':
        # Fit ellipse with SVD -- equivalent to least squares error ellipse, I think? Doesn't seem right though.
        U, S, V = np.linalg.svd(np.vstack(
                (
                    vertices['x'] - center_x,
                    vertices['y'] - center_y
                )
            )
        )

        size = 2.0*np.sqrt(2.0/vertices['x'].shape[0]) # scale factor
        return hv.Ellipse(
                center_x,
                center_y,
                (size*S[0], size*S[1]),
                orientation = np.arctan(U[0][1]/U[0][0])
        )
    if method.lower() == 'lsq':
        xvals = vertices['x'] - center_x
        yvals = vertices['y'] - center_y

        # weight matrix
        COEFF_MATRIX = np.vstack([xvals**2, xvals * yvals, yvals**2, xvals, yvals]).T
        VECTOR = np.ones_like(xvals)

        # Coefficients of the ellipse equation from Wikipedia
        # https://en.wikipedia.org/wiki/Ellipse#Standard_equation
        # Under general ellipse
        A, B, C, D, E = np.linalg.lstsq(COEFF_MATRIX, VECTOR, rcond=None)[0].squeeze()

        width = -2*np.sqrt(2*(A*E**2 + C*D**2 - B*D*E - (B**2 - 4*A*C))*((A+C)+np.sqrt((A-C)**2+B**2)))/(B**2-4*A*C)
        height= -2*np.sqrt(2*(A*E**2 + C*D**2 - B*D*E - (B**2 - 4*A*C))*((A+C)-np.sqrt((A-C)**2+B**2)))/(B**2-4*A*C)
        theta = np.arctan((1.0/B)*(C-A-np.sqrt((A-C)**2 + B**2)))

        return hv.Ellipse(
            center_x,
            center_y,
            (width, height),
            orientation = theta
        )

    raise ValueError(f"Invalid method for ellipse fitting {method}")


def get_largest_polygon_hv(annotation_dict : dict, slice_idx : int = None, n_polygons = 1) -> tuple[hv.element.path.Polygons,int, int]:
    """
    Expects an annotation dict, and returns the largest polygon in it
    + from which slice it came. n_polygons is the number of polygons to return.
    If >1, returns a LIST of tuples, with the 1st tuple being the largest polygon, 
    the next being the next largest polygon, etc. If there
    are fewer polygons than requested, will raise an exception. If slice_idx
    is a list, will return the largest polygon for EACH slice. If slice_idx
    is a list AND n_polygons is a list, will return lists of lists.
    """

    # if it's a list, then return a list.
    if type(slice_idx) is list:
        ret_list = []
        for this_slice in slice_idx:
            if n_polygons == 1:
                roi_idx = np.argmax([polygon_area(p['x'], p['y']) for p in annotation_dict[this_slice]['annotator'].annotated.data])
                if roi_idx is None:
                    raise AssertionError(f"No annotated ROIs in requested image slice indexed as {this_slice}")
                ret_list.append(
                        (
                            annotation_dict[this_slice]['annotator'].annotated.split()[roi_idx], 
                            this_slice,
                            roi_idx
                        )
                    )
            else:
                ret_list.append(
                       _get_largest_polygons_hv(annotation_dict, this_slice, n_polygons) 
                    ) 
        return ret_list

    # if not a list, then return just a single tuple or list of tuples
    if n_polygons > 1:
        return _get_largest_polygons_hv(annotation_dict, slice_idx, n_polygons) # private method for returning more than one. This is me being
        # lazy to avoid a rewrite

    if slice_idx is None:
        largest_poly = {
            slice_idx : max([polygon_area(p['x'],p['y']) for p in annotation_dict[slice_idx]['annotator'].annotated.data])
            for slice_idx in annotation_dict.keys()
            if isinstance(slice_idx, int) and len(annotation_dict[slice_idx]['annotator'].annotated.data) #ignores 'merged'
        }
        slice_idx = max(largest_poly, key = largest_poly.get)
        # do it again.
        roi_idx = np.argmax([polygon_area(p['x'], p['y']) for p in annotation_dict[slice_idx]['annotator'].annotated.data])
    elif slice_idx is int:
        roi_idx = np.argmax([polygon_area(p['x'], p['y']) for p in annotation_dict[slice_idx]['annotator'].annotated.data]) 
    
    return (
        annotation_dict[slice_idx]['annotator'].annotated.split()[roi_idx], 
        slice_idx,
        roi_idx
    )

def _get_largest_polygons_hv(annotation_dict : dict, slice_idx : int = None, n_polygons = 1) -> list[tuple[hv.element.path.Polygons, int, int]]:
    """
    Private method for returning more than one polygon, in order of size. 
    """

    if slice_idx is None:
        # Not Pythonic, who cares
        poly_list = []
        for slice_idx in annotation_dict.keys():
            poly_list += [
                (
                    polygon_area(p['x'],p['y']),
                    slice_idx
                )
                for p in annotation_dict[slice_idx]['annotator'].annotated.data
                if isinstance(slice_idx, int) and len(annotation_dict[slice_idx]['annotator'].annotated.data) # ignore the 'merged' key
            ] # iterate through polygons, store in list
        # now we have a list of tuples that track each polygon

    top_rois = []
    raise NotImplementedError("I haven't finished implementing this yet")

def annotation_dict_to_numpy(annotation_dict : dict, slice_idx : int) -> np.ndarray:
    """ Returns the numpy array underlying the image in a single frame of an annotation dict"""
    return annotation_dict[slice_idx]['layout']['DynamicMap'].I.data[()].Image.I.data['Intensity'] # yeah, I know...