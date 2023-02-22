import numpy as np
from scipy.special import i0
from scipy.optimize import minimize

def cluster_by_correlation(input_frames : np.ndarray, seeds : np.ndarray, kappa: float)->np.ndarray:
    """
    Numpy-only implementation of Hessam's algorithm so that you don't need
    any `SiffPy` or other image processing infrastructure to run it. Uses
    only `numpy`, `scipy.special.i0`, and `scipy.optimize.minimize`.

    Basic outline:

        1. Use a model of multiple von Mises distributions to estimate
        the phase of each seed ROI by matching the correlation matrices
        of the seed ROIs to a correlation matrix of multiple von Mises.
        Each seed ROI then has a corresponding phase: the circular mean
        of its correpsonding von Mises.

        2. Correlate each pixel's time series with the seed ROIs,
        and project the correlations onto a circle by summing the product
        of each pixel's correlation with the seed with exp(i*θ) with θ the
        von Mises phase. This approximates a Fourier transform with the basis
        vectors corresponding to all of the seeds (and the imagined seeds
        which are _not_ sampled).

        3. Cluster the pixels by their phase and amplitude.

    Arguments
    ---------

    input_frames : np.ndarray
    
        timepoints by z slices by y pixels by x pixels
        array corresponding to the full time series of the data to use
        for ROI identification

    seeds : np.ndarray

        z slices by y pixels by x pixels array of dtype = bool 
        corresponding to the pixels of the seed ROIs to use for ROI identification

    kappa : float

        Currently fixes kappa, the concentration parameter of the von Mises
        distributions used to approximate the tuning curves of each seed ROI.
        May make a free parameter in the future.

    Returns
    -------

    px_identity : np.ndarray

        Array of size z slices by y pixels by x pixels of dtype = int,
        corresponding to the ROI identity of each pixel.

    """

    if not (seeds.shape == input_frames.shape[1:]):
        raise ValueError("Seeds and input_frames must have the same shape")

    # Sums each ROI to produce a ROI_seed_count by timepoints array (slow axis is ROI)    
    seed_roi_timeseries = (input_frames @ seeds).T

    # Assigns a phase to every seed
    phases = get_seed_phases(seed_roi_timeseries, kappa = kappa)

    # Approximates the projection of each pixel onto a circle
    corr_fft_approx = corr_fft(phases, seed_roi_timeseries, input_frames)

    # Uses the phase and amplitude of each pixel's projection to
    # assign clusters
    return cluster_by_fft(corr_fft_approx, array_shape = input_frames.shape[1:])

def cluster_by_fft(corr_fft_approx : np.ndarray, array_shape : tuple[int])->np.ndarray:
    """
    Takes the output of `corr_fft` and clusters the pixels by their
    phase and amplitude.

    Arguments
    ---------

    corr_fft_approx : np.ndarray

       (n_pixels,1) array of complex numbers to use to cluster the pixels
       by location, tuning, and tuning strength

    Returns
    -------

    px_identity : np.ndarray

        Array of size z slices by y pixels by x pixels of dtype = int,
        corresponding to the ROI identity of each pixel.
    """
    raise NotImplementedError()

def corr_fft(seed_phases : np.ndarray, seed_roi_timeseries : np.ndarray, input_frames : np.ndarray)->np.ndarray[np.complex128]:
    """
    Takes the correlation of each pixel in input_frames with each seed ROI,
    and projects the correlations onto a circle by summing the product
    of each pixel's correlation with the seed with exp(i*θ) with θ the
    von Mises phase. This approximates a Fourier transform with the basis
    vectors corresponding to all of the seeds (and the imagined seeds
    which are _not_ sampled).
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
    return np.exp(1j*seed_phases).T @ correlation / seed_roi_timeseries.shape[0]
 

def get_seed_phases(seed_roi_timeseries : np.ndarray, kappa : float = 1.0)->list[float]:
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

    phase : float

        Circular mean of the von Mises distribution which best matches
        the seed ROI's correlation matrix.
    """
    # Correlation matrix of the seed ROI
    seed_corr = np.corrcoef(seed_roi_timeseries)

    # Phase for each seed ROI, (ROI_seed_count, 1)
    return match_to_von_mises(seed_corr, kappa = kappa)


def corr_between_von_mises(mu1 : np.ndarray, mu2: np.ndarray, kappa : float = 1.0)->float:
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

def match_to_von_mises(corr_mat : np.ndarray, kappa : float)->np.ndarray:
    """
    Takes a correlation matrix and tries to fit a von Mises
    distribution to each row of the matrix. Returns an array
    of means.
    """
    num_vms = corr_mat.shape[0]
    # uniform around the circle
    initial_mus = np.linspace(0, 2*np.pi, num_vms, endpoint = False)

    # could probably JAX this to make it faster
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
        loss, initial_mus, bounds = [(0, 2*np.pi)] * num_vms,
    )
    return sol.x
