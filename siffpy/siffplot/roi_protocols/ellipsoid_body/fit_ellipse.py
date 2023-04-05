import numpy as np
import logging

import holoviews as hv

from siffpy.siffplot.roi_protocols.roi_protocol import ROIProtocol
from siffpy.siffplot.roi_protocols.ellipsoid_body.extra_rois import ExtraRois
from siffpy.siffplot.roi_protocols.utils.holoviews_fcns import fit_ellipse_to_poly
from siffpy.siffplot.roi_protocols.utils import PolygonSource
from siffpy.siffplot.roi_protocols.rois.ellipse import Ellipse
from siffpy.siffplot.roi_protocols.extern import smallest_circle
# Code for ROI extraction from the ellipsoid body after manual input

class FitEllipse(ROIProtocol):
    name = "Fit ellipse"

    def extract(self, strange_thing : int = 4, stranger_thing : ExtraRois = ExtraRois.CENTER)->Ellipse:
        raise NotImplementedError()
    
    def segment(self):
        raise NotImplementedError()
    pass


def fit_ellipse(
    reference_frames : list,
    polygon_source : PolygonSource,
    *args,
    slice_idx : int = None,
    extra_rois : ExtraRois = ExtraRois.CENTER,
    **kwargs) -> Ellipse:
    """
    Fits the largest polygon drawn in the annotators to an ellipse, 
    and uses that as the outside of the ellipsoid body estimate.

    Seems unlikely to use this if you're not using holoviews, but
    it's here for completeness.

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


    largest_polygon, slice_idx, roi_idx = polygon_source.get_largest_polygon(slice_idx = slice_idx)
    source_image = polygon_source.source_image(slice_idx)

    ellip = fit_ellipse_to_poly(largest_polygon, method = 'lsq')

    center_x, center_y = ellip.x, ellip.y
    center_poly = None

    if (
        (polygon_source.interface.value == 'Holoviews') and
        (len(polygon_source.polygons(slice_idx=slice_idx)) > 1)
    ):

        # there may be other ROIs here with additional information
        if not ((extra_rois is ExtraRois.NONE) or (extra_rois == ExtraRois.NONE.value)):
            if ((extra_rois is ExtraRois.CENTER) or (extra_rois == ExtraRois.CENTER.value)):
                # Go through all polygons, find the one with a center closest to the largest polygon
                centers = []
                polys = polygon_source.polygons(slice_idx=slice_idx)
                for poly in polys:
                    (c_x, c_y, _) = smallest_circle.make_circle(list(zip(poly['x'],poly['y'])))
                    centers.append((c_x,c_y))
                
                dists = len(centers)*[np.nan]
                for c_idx in range(len(centers)):
                    if c_idx == roi_idx:
                        continue
                    dists[c_idx] = (centers[c_idx][0] - center_x)**2 + (centers[c_idx][1] - center_y)**2
                
                nearest_poly_idx = np.nanargmin(dists)
                
                #reassign the ellipse center to this smallest polygon's center.
                annotation_dict = polygon_source.source
                center_poly = annotation_dict[slice_idx]['annotator'].annotated.split()[nearest_poly_idx]

            else:
                logging.warning(f"""
                    Invalid argument {kwargs['extra_rois']} provided for extra_rois keyword. Ignoring.
                """)
        else:
            # was not in kwargs
            logging.warning("Additional polygons detected with no useful argument for 'extra_rois'. Ignoring polygons")

    orientation = 0.0
    try:
        orientation = polygon_source.orientation
    except NotImplementedError:
        pass
    # If you're using napari instead
    else:
        Ellipse(
            ellip,
            source_polygon = largest_polygon,
            center_poly = center_poly,
            slice_idx = slice_idx,
            image = source_image,
            name = 'Ellipsoid body',
            orientation = orientation,
        )
        logging.warning("Haven't gotten around to implementing the center ROI stuff fit to a Polygon in napari.")
    
    return Ellipse(
        ellip.opts(**kwargs),
        source_polygon = largest_polygon,
        center_poly = center_poly,
        slice_idx = slice_idx,
        image = source_image,
        name = 'Ellipsoid body',
        orientation = orientation
    )