# Code for ROI extraction from the protocerebral bridge after manual input
import numpy as np
from holoviews import Polygons
from scipy.special import i0

from siffpy import SiffReader
from siffpy.siffplot.roi_protocols import rois, ROIProtocol
from siffpy.siffplot.roi_protocols.utils import (
    PolygonSource, polygon_to_mask, polygon_to_z
)
from siffpy.siffplot.roi_protocols.protocerebral_bridge.von_mises.numpy_implementation import (
    match_to_von_mises
)
from siffpy.siffplot.roi_protocols.protocerebral_bridge.von_mises.napari_tools import (
    CorrelationWindow
)

class FitVonMises(ROIProtocol):

    name = "Fit von Mises"
    base_roi_text = "View correlation map"

    def on_click(self, segmentation_widget):
        corr_window = CorrelationWindow(segmentation_widget)

    def extract(
            self,
            reference_frames : np.ndarray,
            polygon_source : PolygonSource,
    )->rois.GlobularMustache:
        return fit_von_mises(
            reference_frames,
            polygon_source,
            #slice_idx=slice_idx,
            #extra_rois=extra_rois,
        )

    def segment(self):
        raise NotImplementedError()

def fit_von_mises(
        reference_frames : list,
        polygon_source : PolygonSource,
        siffreader : SiffReader,
        *args,
        n_glomeruli : int = 16,
        slice_idx : int = None,
        kappa : float = 1.0, # kappa is the concentration parameter of the von Mises distribution
        timepoint_lower_bound : int = 0,
        timepoint_upper_bound : int = None,
        **kwargs
    )-> rois.GlobularMustache:
    """
    Uses Hessam's algorithm to try to fit an n_glomerulus vector
    presuming each has a von Mises distributed tuning curve with
    mean distributed uniformly around the circle.
    """
    if timepoint_upper_bound == 0:
        timepoint_upper_bound = siffreader.im_params.shape[0]

    # Gets all polygons, irrespective of slice
    seed_polys = polygon_source.polygons()

    # Compute the correlation matrix between
    # each profile across the time series
    corr_mat, seed_time_series = polys_to_corr(
        seed_polys,
        siffreader,
        timepoint_lower_bound=timepoint_lower_bound,
        timepoint_upper_bound=timepoint_upper_bound
    )

    # Compute a von Mises model with the same
    # number of seeds that matches the correlation
    # matrix between the true data seeds

    v_means = match_to_von_mises(corr_mat)

    # Take the FFT of each pixel's correlation against
    # the seeds and extract a phase (from the fundamental
    # frequency) of the FFT

    framelist = siffreader.im_params.flatten_by_timepoints(
        timepoint_start = timepoint_lower_bound,
        timepoint_end = timepoint_upper_bound
    )

    # Flattened
    frame_array = np.array(
        siffreader.get_frames(
            frames = framelist,
            registration_dict = siffreader.registration_dict if hasattr(siffreader, 'registration_dict') else None
        )
    ).reshape((-1, np.prod(siffreader.im_params.stack[1:])))

    zscored_frames = (frame_array - frame_array.mean(axis=0)) / frame_array.std(axis=0)
    zscored_seed = (seed_time_series - seed_time_series.mean(axis=0)) / seed_time_series.std(axis=0)

    pxwise_fft = (np.exp(1j*v_means) @ (zscored_frames.T @ zscored_seed)).reshape(siffreader.im_params.stack[1:])


    raise NotImplementedError("Sorry!")


def von_mises_corr(mu1 : float, mu2: float, kappa : float, noise_var : float = 0.0)->float:
    """
    Computes the correlation between two von Mises
    distributions with means mu1 and mu2 and circular
    variance kappa. Noise var is the variance of random noise
    applied to all measurements (presumed to be independent
    and mean-zero! Both assumptions are not likely to be true:
    noise will usually come from motion, autofluorescence, etc.
    which is both positive mean _and_ correlated).

    Formula for the von Mises (relatively easy to derive):

    I0( 2κ cos((μ2-μ1)/2)) ) - I0(κ)^2

    --------------------------------
    
        I0(2κ) - I0(κ)^2

    where I0(x) is the modified Bessel function of the first kind of order 0,
    defined as (1/2π) times the integral from -π to +π of exp(κ cos(t))dt.

    An independent source of noise simply adds to the variance of each
    individual von Mises without affecting their covariance. 
    """
    #var_1 = 
    vm_var = i0(2*kappa) - i0(kappa)**2
    vm_corr = (
        (i0(2*kappa*np.cos((mu2-mu1)/2)) - i0(kappa)**2)/
        vm_var
    )

    #noise_factor = np.sqrt(1+2*noise_var/vm_var + (noise_var/vm_var)**2)

    #Faster, approximate though
    noise_factor = 1 + noise_var/vm_var

    return vm_corr/noise_factor

def polys_to_corr(
        polys : list[Polygons],
        siffreader : SiffReader,
        timepoint_lower_bound : int = 0,
        timepoint_upper_bound : int = None,
        dtype : type = np.uint64
    )->tuple[np.ndarray, np.ndarray]:
    """
    Computes the correlation matrix between each polygon
    and the time series of the siffreader
    """

    masks = [polygon_to_mask(poly, siffreader.im_params.shape) for poly in polys]
    zs = [polygon_to_z(poly) for poly in polys]
    if timepoint_upper_bound is None:
        timepoint_upper_bound = siffreader.im_params.num_timepoints
    seed_t_series : np.ndarray = np.array([
        siffreader.sum_mask(
            mask,
            z_list = z if isinstance(z, list) else [z],
            timepoint_start = timepoint_lower_bound,
            timepoint_end = timepoint_upper_bound
        )
        for mask, z in zip(masks,zs)
    ],dtype=dtype)

    return (np.corrcoef(seed_t_series), seed_t_series)