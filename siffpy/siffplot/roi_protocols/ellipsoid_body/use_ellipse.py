import numpy as np
import logging

import holoviews as hv

from siffpy.siffplot.roi_protocols.ellipsoid_body.extra_rois import ExtraRois
from siffpy.siffplot.roi_protocols.utils import PolygonSource
from siffpy.siffplot.roi_protocols.rois.ellipse import Ellipse
from siffpy.siffplot.roi_protocols.extern import smallest_circle


def use_ellipse(
    reference_frames : list,
    polygon_source : PolygonSource,
    *args,
    slice_idx : int = None,
    extra_rois : ExtraRois = ExtraRois.CENTER,
    **kwargs) -> Ellipse:
    """
    Simply takes the largest ellipse type shape in a viewer
    and uses it as the bound! polygon_source has to be a
    Viewer or an object that can be treated like one
    (e.g. NapariInterface), since HoloViews doesn't have
    a way to annotate with an ellipse.

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
    
    ellip, slice_idx, roi_idx = polygon_source.get_largest_ellipse(slice_idx = slice_idx)
    source_image = polygon_source.source_image(slice_idx = slice_idx)

    center_x, center_y = ellip.x, ellip.y
    center_poly = None

    if not ((extra_rois is ExtraRois.NONE) or (extra_rois == ExtraRois.NONE.value)):
        
        viable_polygons = polygon_source.polygons(slice_idx=slice_idx)
        if len(viable_polygons) < 2:
            raise ValueError("Did not provide a second ROI for the extra ROI field")
        
        if (extra_rois is ExtraRois.CENTER or (extra_rois == ExtraRois.CENTER.value)):
            # Go through all polygons, find the one with a center closest to the largest polygon
            if not (slice_idx is None):
                viable_polygons = [
                    shape for shape in viable_polygons
                    if int(np.round(shape[0,0])) == slice_idx
                ]

            centers = []
            for shape in viable_polygons:
                (c_x, c_y, _) = smallest_circle.make_circle(list(zip(shape[:,-1],shape[:,-2])))

                if slice_idx is None:
                    centers.append((c_x,c_y))
                else:
                    z_plane = int(np.round(shape[0,0]))
                    if slice_idx == z_plane:
                        centers.append((c_x,c_y))
            
            dists = len(centers)*[np.nan]
            for c_idx in range(len(centers)):
                if c_idx == roi_idx: # but not the largest polygon itself!
                    continue
                dists[c_idx] = (centers[c_idx][0] - center_x)**2 + (centers[c_idx][1] - center_y)**2
            
            nearest_poly_idx = np.nanargmin(dists)
            
            #reassign the ellipse center to this smallest polygon's center.
            center_poly_array = viable_polygons[nearest_poly_idx]
            center_poly = hv.Polygons(
                {
                    ('y', 'x') : center_poly_array[:,-2:]
                }
            )

        else:
            logging.warning(f"""
                Invalid argument {extra_rois} provided for extra_rois keyword. Ignoring.
            """)

    orientation = 0.0
    try:
        orientation = polygon_source.orientation
    except NotImplementedError:
        pass

    return Ellipse(
        ellip,
        source_polygon = None,
        center_poly = center_poly,
        slice_idx = slice_idx,
        image = source_image,
        name = 'Ellipsoid body',
        orientation = orientation
    )