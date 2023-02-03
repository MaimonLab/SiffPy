# Code for ROI extraction from the protocerebral bridge after manual input

from siffpy import SiffReader

from siffpy.siffplot.roi_protocols.utils import PolygonSource
from siffpy.siffplot.roi_protocols import rois

def fit_von_mises(
        reference_frames : list,
        polygon_source : PolygonSource,
        siffreader : SiffReader,
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
    im_params = siffreader.im_params

    print(im_params)

    # Gets all polygons, irrespective of slice
    seed_polys = polygon_source.polygons()

    # Compute the correlation matrix between
    # each profile across the time series

    # Compute a von Mises model with the same
    # number of seeds that matches the correlation
    # matrix between the true data seeds

    # Take the FFT of each pixel's correlation against
    # the seeds and extract a phase (from the fundamental
    # frequency) of the FFT

    raise NotImplementedError("Sorry!")