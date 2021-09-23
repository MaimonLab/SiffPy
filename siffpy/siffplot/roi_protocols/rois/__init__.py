# Shared code for ROI extraction

import numpy as np
import holoviews as hv
from ..extern.smallest_circle import make_circle
from .roi import *
from .ellipse import *

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

def get_largest_polygon(annotation_dict : dict, slice_idx : int = None) -> tuple[hv.element.path.Polygons,int, int]:
    """
    Expects an annotation dict, and returns the largest polygon in it + from which slice it came
    """
    if slice_idx is None:
        largest_poly = {
            slice_idx : max([polygon_area(p['x'],p['y']) for p in annotation_dict[slice_idx]['annotator'].annotated.data])
            for slice_idx in annotation_dict.keys()
            if isinstance(slice_idx, int) and len(annotation_dict[slice_idx]['annotator'].annotated.data) #ignores 'merged'
        }
        slice_idx = max(largest_poly, key = largest_poly.get)
        # do it again.
        roi_idx = np.argmax([polygon_area(p['x'], p['y']) for p in annotation_dict[slice_idx]['annotator'].annotated.data])
    else:
        roi_idx = np.argmax([polygon_area(p['x'], p['y']) for p in annotation_dict[slice_idx]['annotator'].annotated.data]) 
    
    return (
        annotation_dict[slice_idx]['annotator'].annotated.split()[roi_idx], 
        slice_idx,
        roi_idx
    )