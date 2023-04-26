from typing import Union
from dataclasses import dataclass

import numpy as np
from scipy.special import i0
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2
import scipy.ndimage as ndi

@dataclass
class VonMisesFit():
    """ Stores the parameters of a von Mises distribution """
    mean : float
    kappa : float

    def pdf(self, theta : np.ndarray)->np.ndarray:
        return np.exp(self.kappa*np.cos(theta - self.mean))/(2*np.pi*i0(self.kappa))

class VonMisesCollection():
    """
    Holds a list of von Mises fits and provides
    acces for a few useful functions
    """

    def __init__(self, vms : list[VonMisesFit]):
        self.vms = vms

    #Get from the list of von Mises fits
    def __getitem__(self, key):
        return self.vms.__getitem__(key)
    
    def __len__(self)->int:
        return len(self.vms)

    @property
    def means(self)->np.ndarray:
        return np.array([vm.mean for vm in self.vms])
    
    @property
    def kappas(self)->np.ndarray:
        return np.array([vm.kappa for vm in self.vms])
    
    @property
    def cov(self)->np.ndarray:
        return corr_between_von_mises(
            self.means,
            self.means,
            self.kappas[0],
        )


def corr_fft(
        seed_phases : np.ndarray,
        seed_roi_timeseries : np.ndarray,
        input_frames : np.ndarray
    )->np.ndarray[np.complex128]:
    """
    Takes the correlation of each pixel in input_frames with each seed ROI,
    and projects the correlations onto a circle by summing the product
    of each pixel's correlation with the seed with exp(i*θ) with θ the
    von Mises phase. This approximates a Fourier transform with the basis
    vectors corresponding to all of the seeds (and the imagined seeds
    which are _not_ sampled).

    Arguments
    ----------

    seed_phases : np.ndarray

        (ROI_seed_count,1) array of complex numbers corresponding to the
        phase of each seed ROI (in radians!)

    seed_roi_timeseries : np.ndarray

        ROI_seed_count by timepoints array of values corresponding to the
        sum of each seed ROI's pixel values.

    input_frames : np.ndarray

        timepoints by y pixels by x pixels array of values corresponding
        to the pixel values of each frame (intensity only).
    
    Returns
    -------

    px_fft : np.ndarray

        n_pixels array of complex numbers relaying the phase and
        sine-ness of each pixel's correlation with the seed ROIs.
    
    """

    # Product - product of means
    # --------------------------
    # Product of std. dev. of both

    # Yields a ROI_seed_count by pixel_number array of values
    # from -1 to 1, corresponding to each pixel's correlation with
    # each seed ROI

    correlation = (
        ( # numerator
            seed_roi_timeseries @ input_frames/input_frames.shape[0] - #E[XY]
            np.outer( # E[X]E[Y]
                seed_roi_timeseries.mean(axis = 1),
                input_frames.mean(axis = 0)
            )
        ) / ( # denominator
            np.outer( # sqrt(Var[X]Var[Y])
                seed_roi_timeseries.std(axis = 1),
                input_frames.std(axis = 0)
            )
        )
    )

    # n_pixels array of complex numbers relaying the phase and
    # sine-ness of each pixel's correlation with the seed ROIs
    # (i.e. fundamental frequency in FFT). Note that if they're
    # _truly_ von Mises, they should not have a pure sinusoid
    # correlation -- it should instead be a rescaled I0(2*kappa*cos(theta - theta_0))
    # with the DC offset removed (I0(kappa)**2) But this is fairly
    # close to a sinusoid for small kappa!

    # The "correct" phase to use would invert the I0 function and then
    # take the arccos of the result. For kappa near 1, this is approximately
    # 2*np.sqrt(2*kappa*cos(Delta_theta/2)-1) = I0(kappa)**2 + corr*(I0(2k)**2 - I0(k)**2)
    return np.exp(1j*seed_phases).T @ correlation / seed_phases.shape[0]
 

def get_seed_phases(
        seed_roi_timeseries : np.ndarray,
        kappa : float = 1.0
    )->tuple[list[VonMisesFit],np.ndarray]:
    """
    Computes the phases of seed ROIs by matching the correlation
    matrix of the seed ROI to a correlation matrix of multiple von Mises
    distributions. Returns the circular mean of the von Mises distribution
    which best matches the seed ROI's correlation matrix.

    Arguments
    ---------

    seed_roi_timeseries : np.ndarray

        timepoints by 1 array of the time series of the seed ROI

    kappa : float

        Concentration parameter of the von Mises distributions used to
        approximate the tuning curves of each seed ROI. May make a free
        parameter in the future.

    Returns
    -------

    phases, covariance : tuple[np.ndarray, np.ndarray]

        First element, phases :
        Circular mean of the von Mises distribution which best matches
        the seed ROI's correlation matrix.

        Second element, covariance :
        Covariance matrix of the seed ROI's correlation matrix.
    """
    # Correlation matrix of the seed ROI
    seed_corr = np.corrcoef(seed_roi_timeseries)

    # Phase for each seed ROI, (ROI_seed_count, 1)
    return (
        VonMisesCollection(match_to_von_mises(seed_corr, kappa = kappa)),
        seed_corr
    )

def corr_between_von_mises(
        mu1 : np.ndarray,
        mu2: np.ndarray,
        kappa : float = 1.0
    )->np.ndarray:
    """
    Computes the correlation between two von Mises
    distributions with means mu1 and mu2 and circular
    variance kappa. Does NOT use noise estimate -- 
    can easily be improved to do so (just a factor in the
    numerator and denominator)

    Formula (relatively easy to derive):

    I0( 2κ cos((μ2-μ1)/2)) ) - I0(κ)^2

    --------------------------------
    
        I0(2κ) - I0(κ)^2

    where I0(x) is the modified Bessel function of the first kind of order 0,
    defined as (1/2π) times the integral from -π to +π of exp(κ cos(t))dt
    """
    return (
        (i0(2*kappa*np.cos(np.subtract.outer(mu2,mu1)/2)) - i0(kappa)**2)/
        (i0(2*kappa) - i0(kappa)**2)
    )

def match_to_von_mises(corr_mat : np.ndarray, kappa : float)->list[VonMisesFit]:
    """
    Takes a correlation matrix and tries to fit a von Mises
    distribution to each row of the matrix. Returns an array
    of means.
    """
    num_vms = corr_mat.shape[0]
    # uniform around the circle
    initial_mus = np.linspace(0, 2*np.pi, num_vms, endpoint = False)
    bounds = [(0, 2*np.pi)] * num_vms
    if kappa is None:
        initial_mus = np.append(initial_mus, 2.0) # initial kappa
        bounds.append((0, np.inf))
        # loss func def
        def loss(mus_and_kappa : np.ndarray):
            return np.sum(
                (
                    corr_mat -
                    corr_between_von_mises(
                        mus_and_kappa[:-1],
                        mus_and_kappa[:-1],
                        kappa = mus_and_kappa[-1],
                    )
                )** 2
            )
    else:
        # could probably numba this to make it faster
        def loss(mus : np.ndarray):
            return np.sum(
                ( corr_mat -
                corr_between_von_mises(mus, mus, kappa = kappa)
                ) ** 2
            )
    
    # TODO maybe there are faster ways, e.g. use the derivative matrix
    # (easy to compute, basically just I1(2*kappa*cos((mu2-mu1)/2)) )
    # But this seems fast
    
    sol = minimize(
        loss, initial_mus, bounds = bounds,
    )
    if kappa is None:
        return [VonMisesFit(mu, sol.x[-1]) for mu in sol.x[:-1]]
    return [VonMisesFit(mu, kappa) for mu in sol.x]
