# Code for ROI extraction from the protocerebral bridge after manual input

from siffpy import SiffReader

from siffpy.siffplot.roi_protocols.utils import PolygonSource
from siffpy.siffplot.roi_protocols import rois

def fit_von_mises(
        reference_frames : list,
        polygon_source : PolygonSource,
        *args,
        n_glomeruli : int = 16,
        slice_idx : int = None,
        circular_variance : float = 1.0,
        **kwargs
    )-> rois.GlobularMustache:
    """
    Uses Hessam's algorithm to try to fit an n_glomerulus vector
    presuming each has a von Mises distributed tuning curve with
    mean distributed uniformly around the circle.
    """

    # Get the siffreader to acquire the actual
    # individual time frames
    siffreader = SiffReader() #= get_siffreader()
    im_params = siffreader.im_params

    # Gets all polygons, irrespective of slice
    seed_polys = polygon_source.polygons()

    # Iterate over polygons,
    # take the correlation between the image stack
    # and its presumed activity profile, and use
    # that as a mask?

    # Sort polygons by presumed peak tuning

    # Take FFT of the correlations

    # Initializes kernels

    #???


    raise NotImplementedError("Sorry!")