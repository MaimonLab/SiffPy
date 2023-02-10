# Code for ROI extraction from the protocerebral bridge after manual input
import numpy as np
from holoviews import Polygons

from siffpy import SiffReader
from siffpy.siffplot.roi_protocols.utils import PolygonSource, polygon_to_mask, polygon_to_z
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
    mean distributed uniformly around the circle. Pseudo-code:

    rois : list[ROI] = select_seed_rois() # k seed ROIs

    def order_rois(rois : list[ROI])->list[ROI]:
        
        seed_corr : np.ndarray = correlation(rois, rois)
        
        vonMisesParams : list[tuple[float, float]] = match_correlation_matrix_to_von_Mises(seed_corr) # k von Mises
        
        _, indices = sort(vonMisesParams, by = 'circular_mean') # orders the von Mises distributions by mean and        figure out how to order them -- the von Mises are just to parameterize the ROIs by a `mean`
        
        return rois[indices] # reorders the ROI list so that they're ordered around a circle, instead of random ordering

    rois = order_rois(rois)

    intensity : np.ndarray = get_all_frames() # t timepoints by n pixels
    corr = correlation(intensity, rois) # returns an n by k matrix of correlations with each seed ROI, now ordered around a circular so that you can take the FFT meaningfully

    ftt_corr = fft(corr, axis = 1) # each pixel's correlation with the ROIs' FFT along the circle
    fundamental_freq : np.ndarray[np.complex] = fft_corr[:,1] # the frequency component that completes one full revolution along the circle, an n by 1 vector
    # each element of fundamental_freq is a complex number whose amplitude reflects how periodic its correlation matrix is and whose phase corresponds to its alignment to the seed ROIs

    new_rois = cluster_phases(fundamental_freq)

    new_rois = order_rois(new_rois)
    """

    # Gets all polygons, irrespective of slice
    seed_polys = polygon_source.polygons()

    # Compute the correlation matrix between
    # each profile across the time series
    polys_to_corr(seed_polys, siffreader)


    # Compute a von Mises model with the same
    # number of seeds that matches the correlation
    # matrix between the true data seeds

    # Take the FFT of each pixel's correlation against
    # the seeds and extract a phase (from the fundamental
    # frequency) of the FFT

    raise NotImplementedError("Sorry!")

def polys_to_corr(polys : list[Polygons], siffreader : SiffReader)->np.ndarray:
    """
    Computes the correlation matrix between each polygon
    and the time series of the siffreader
    """

    masks = [polygon_to_mask(poly, siffreader.im_params.shape) for poly in polys]
    zs = [polygon_to_z(poly) for poly in polys]
    seed_t_series : np.ndarray = np.array([
        siffreader.sum_mask(
            mask,
            z_list = z
        )
        for mask, z in zip(masks,zs)
    ])

    return np.corrcoef(seed_t_series)