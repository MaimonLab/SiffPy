import numpy as np
import logging
from enum import Enum

import holoviews as hv

from . import rois
from .extern import smallest_circle
# Code for ROI extraction from the ellipsoid body after manual input

class ExtraRois(Enum):
    CENTER : str = 'center'
    NONE : str = 'none'

def fit_ellipse(reference_frames : list, polygon_source : dict,
    *args, slice_idx : int = None, extra_rois : ExtraRois = ExtraRois.CENTER, **kwargs) -> rois.Ellipse:
    """
    Fits the largest polygon drawn in the annotators to an ellipse, 
    and uses that as the outside of the ellipsoid body estimate.

    Keyword args
    ------------
    slice_idx : int

        Takes the largest polygon only from within
        the slice index labeled 'slice_idx', rather
        than the largest polygon across all slices.

    extra_rois : str | ExtraRois

        A string that explains what any ROIs other than
        the largest might be useful for. Current options:

            - Center : Finds the extra ROI "most likely" to 
            be the center of the ellipse, and uses it to reshape the
            fit polygon. Currently, that ROI is just the smallest 
            other one... Not even constrained to be fully contained.

    Additional kwargs are passed to the Ellipse's opts function

    """
    ## Outline
    #
    #   Search through the annotated polygons (whether in napari or holoviews)
    #   and identify the largest. Fit it to an ellipse, then use that ellipse to
    #   produce an Ellipse class. If there are other polygons, and the keyword argument
    #   is provided, use those other polygons to elaborate on the ellipse (e.g. to highlight the center).
    #   TODO: IMPLEMENT USING AN ELLIPSE SHAPE IN NAPARI, AND ALSO THE CENTER HOLE ARGUMENT

    using_holoviews = True
    if not (type(polygon_source) is dict):
        using_holoviews = False

    if using_holoviews:
        annotation_dict = polygon_source
        largest_polygon, slice_idx, roi_idx = rois.get_largest_polygon_hv(annotation_dict, slice_idx = slice_idx)
        source_image = rois.annotation_dict_to_numpy(annotation_dict,slice_idx)
    else:
        largest_polygon, slice_idx, roi_idx = rois.get_largest_polygon_napari(polygon_source, shape_layer_name = 'ROI shapes', slice_idx = slice_idx)
        reference_frame_layer = next(filter(lambda x: x.name == 'Reference frames', polygon_source.layers)) # get the layer with the reference frames
        source_image = reference_frame_layer.data[slice_idx, :, :]

    ellip = rois.fit_ellipse_to_poly(largest_polygon, method = 'lsq')

    center_x, center_y = ellip.x, ellip.y
    center_poly = None

    if using_holoviews:
        if len(annotation_dict[slice_idx]['annotator'].annotated.data) > 1:
            # there may be other ROIs here with additional information
            if not ((extra_rois is ExtraRois.NONE) or (extra_rois == ExtraRois.NONE.value)):
                if ((extra_rois is ExtraRois.CENTER) or (extra_rois == ExtraRois.CENTER.value)):
                    # Go through all polygons, find the one with a center closest to the largest polygon
                    centers = []
                    for poly in annotation_dict[slice_idx]['annotator'].annotated.data:
                        (c_x, c_y, _) = smallest_circle.make_circle(list(zip(poly['x'],poly['y'])))
                        centers.append((c_x,c_y))
                    
                    dists = len(centers)*[np.nan]
                    for c_idx in range(len(centers)):
                        if c_idx == roi_idx:
                            continue
                        dists[c_idx] = (centers[c_idx][0] - center_x)**2 + (centers[c_idx][1] - center_y)**2
                    
                    nearest_poly_idx = np.nanargmin(dists)
                    
                    #reassign the ellipse center to this smallest polygon's center.
                    center_poly = annotation_dict[slice_idx]['annotator'].annotated.split()[nearest_poly_idx]

                else:
                    logging.warning(f"""
                        Invalid argument {kwargs['extra_rois']} provided for extra_rois keyword. Ignoring.
                    """)
            else:
                # was not in kwargs
                logging.warning("Additional polygons detected with no useful argument for 'extra_rois'. Ignoring polygons")
    
    # If you're using napari instead
    else:
        rois.Ellipse(
            ellip,
            source_polygon = largest_polygon,
            center_poly = center_poly,
            slice_idx = slice_idx,
            image = source_image,
            name = 'Ellipsoid body'
        )
        logging.warning("Haven't gotten around to implementing the center ROI stuff fit to a Polygon in napari.")
    
    return rois.Ellipse(
        ellip.opts(**kwargs),
        source_polygon = largest_polygon,
        center_poly = center_poly,
        slice_idx = slice_idx,
        image = source_image,
        name = 'Ellipsoid body'
    )

def use_ellipse(reference_frames : list, polygon_source,
    *args, slice_idx : int = None, extra_rois : ExtraRois = ExtraRois.CENTER, **kwargs) -> rois.Ellipse:
    """
    Simply takes the largest ellipse type shape in a viewer
    and uses it as the bound! polygon_source has to be a
    Viewer or an object that can be treated like one
    (e.g. NapariInterface)

    Keyword args
    ------------
    slice_idx : int

        Takes the largest polygon only from within
        the slice index labeled 'slice_idx', rather
        than the largest polygon across all slices.

    extra_rois : str | ExtraRois

        A string that explains what any ROIs other than
        the largest might be useful for. Current options:

            - Center : Finds the extra ROI "most likely" to 
            be the center of the ellipse, and uses it to reshape the
            fit polygon. Currently, that ROI is just the smallest 
            other one... Not even constrained to be fully contained.

    Additional kwargs are passed to the Ellipse's opts function

    """
    ## Outline
    #
    #   Search through the annotated polygons (whether in napari or holoviews)
    #   and identify the largest. Fit it to an ellipse, then use that ellipse to
    #   produce an Ellipse class. If there are other polygons, and the keyword argument
    #   is provided, use those other polygons to elaborate on the ellipse (e.g. to highlight the center).
    #   TODO: IMPLEMENT USING AN ELLIPSE SHAPE IN NAPARI, AND ALSO THE CENTER HOLE ARGUMENT
    
    ellip, slice_idx, roi_idx = rois.get_largest_ellipse_napari(polygon_source, shape_layer_name = 'ROI shapes', slice_idx = slice_idx)
    reference_frame_layer = next(filter(lambda x: x.name == 'Reference frames', polygon_source.layers)) # get the layer with the reference frames
    source_image = reference_frame_layer.data[slice_idx, :, :]

    center_x, center_y = ellip.x, ellip.y
    center_poly = None

    if not ((extra_rois is ExtraRois.NONE) or (extra_rois == ExtraRois.NONE.value)):
        
        poly_layer = next(filter(lambda x: x.name == 'ROI shapes', polygon_source.layers),None) 
        
        if len(poly_layer.data) < 2:
            pass
        else: # at least two shapes available
            if (extra_rois is ExtraRois.CENTER or (extra_rois == ExtraRois.CENTER.value)):
                # Go through all polygons, find the one with a center closest to the largest polygon
                centers = []
                for shape in poly_layer.data:
                    (c_x, c_y, _) = smallest_circle.make_circle(list(zip(shape[:,-1],shape[:,-2])))
                    centers.append((c_x,c_y))
                
                dists = len(centers)*[np.nan]
                for c_idx in range(len(centers)):
                    if c_idx == roi_idx:
                        continue
                    dists[c_idx] = (centers[c_idx][0] - center_x)**2 + (centers[c_idx][1] - center_y)**2
                
                nearest_poly_idx = np.nanargmin(dists)
                
                #reassign the ellipse center to this smallest polygon's center.
                center_poly_array = poly_layer.data[nearest_poly_idx]
                center_poly = hv.Polygons(
                    {
                        ('y', 'x') : center_poly_array[:,-2:]
                    }
                )

            else:
                logging.warning(f"""
                    Invalid argument {extra_rois} provided for extra_rois keyword. Ignoring.
                """)
    
    return rois.Ellipse(
        ellip,
        source_polygon = None,
        center_poly = center_poly,
        slice_idx = slice_idx,
        image = source_image,
        name = 'Ellipsoid body'
    )

def dummy_method(*args, **kwargs):
    print("I'm just a placeholder!")