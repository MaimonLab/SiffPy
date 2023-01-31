import warnings, random

import numpy as np
from numpy import fft
from numpy.linalg import solve
import scipy
import scipy.ndimage
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from siffpy.core.utils.circle_fcns import *
from siffreadermodule import SiffIO

def build_reference_image(siffio : SiffIO, frames : list[int], ref_method : str = 'suite2p', **kwargs) -> np.ndarray:
    """
    Constructs a reference image for alignment. WARNING: DOES NOT TYPECHECK FOR COMPATIBLE SHAPES

    INPUTS
    ------
    frames : list[int]

        list of the integer indices of the frames to use to build the reference image.

    ref_method : str = 'average'

        What method to use to construct the reference image. Options are:
            'average' -- Take the average of the frames
            'suite2p' -- Iterative alignment of random frames + averaging

    kwargs : dict

        KWARGS are used only for suite2p reference images. Use the help docstring
        for siffpy.core.utils.registration.suite2p_reference to learn more.

    RETURNS
    -------
    np.ndarray:

        A reference image built from the input frames
    """

    if ref_method == 'average':
        return siffio.pool_frames(pool_lists=[frames], flim = False)

    if ref_method == 'suite2p':
        return suite2p_reference(siffio, frames, **kwargs)

    raise TypeError("Method passed for constructing average not valid")

def suite2p_reference(siffio : SiffIO, frames : list[int], **kwargs) -> np.ndarray:
    """
    Implements the alignment procedure used by suite2p, iteratively averaging
    the frames most correlated with all other frames in a random subset.

    Modeled off the description given at https://suite2p.readthedocs.io/en/latest/registration.html

    INPUTS
    ------

    frames : list[int]

        List of the frames to use to compute a reference

    nimg_init (optional): int

        Number of randomly subsampled frames from the argument
        'frames' to use for initial iterative alignment. Default
        is min(len(frames)/10,200)

    seed_ref_count (optional) : int

        Number of most-correlated frames to use to construct a
        reference image. A smaller number will have sharper boundaries
        for alignment, but will take a noisier reference. A larger
        number will have fuzzier boundaries that are more reliable.
        Must be smaller than nimg_init. Default is 20 (like suite2p)

    registration_dict (optional) : dict

        Lookup table to align the frames (used durign iterative alignment)

    """

    nimg_init = min(int(len(frames)/10), 200)
    if 'nimg_init' in kwargs:
        nimg_init = kwargs['nimg_init']
        if not isinstance(nimg_init, int):
            warnings.warn("Suite2p alignment arg 'nimg_init' is not of type int. "
                          "Using default value instead.")
            nimg_init = min(int(len(frames/10)),200)
        if nimg_init > len(frames):
            warnings.warn("Suite2p alignment arg 'nimg_init' is greater than number "
                          f"of frames being aligned. Defaulting to {len(frames)}.")
            nimg_init = len(frames)

    registration_dict = {}
    if 'registration_dict' in kwargs:
        registration_dict = kwargs['registration_dict']
        if not isinstance(registration_dict, dict):
            warnings.warn("Suite2p alignment arg 'registration_dict' is not of type dict."
                          " Using an empty dict instead")
            registration_dict = {}

    # randomly sample nimg_init frames    

    init_frames_idx = random.sample(frames, nimg_init)

    if 'discard_bins' in kwargs:
        discard_bins = None
        if isinstance(kwargs['discard_bins'], int):
            discard_bins = kwargs['discard_bins']

    init_frames = np.array(
                    siffio.get_frames(
                        frames = init_frames_idx, 
                        flim = False,
                        registration = registration_dict
                    )
                  )
    
    # find the few frames most correlated with the mean of these

    mean_img = np.nanmean(init_frames,axis=0)
    mean_subbed  = (init_frames - mean_img)**2
    err_val = np.sum(mean_subbed,axis=(1,2))

    # how many of the most correlated frames to take
    seed_ref_count = 100
    if 'seed_ref_count' in kwargs:
        seed_ref_count = kwargs['seed_ref_count']
        if not isinstance(seed_ref_count, int):
            seed_ref_count = 100
            warnings.warn("Suite2p alignment arg 'seed_ref_count' is not of type int. "
                          f"Using {seed_ref_count} instead.")
    if seed_ref_count > nimg_init:
        seed_ref_count = nimg_init - 1
        warnings.warn("Suite2p alignment arg 'seed_ref_count' is greater than number "
                        f"of frames being aligned. Defaulting to {seed_ref_count}.")
            
    seed_idx = np.argpartition(err_val, seed_ref_count)[:seed_ref_count]
    return np.squeeze(np.mean(init_frames[seed_idx,:,:],axis=0))

def align_to_reference(ref : np.ndarray, im : np.ndarray, shift_only : bool = False, 
                       subpx : bool = False, phase_corr : bool = False, blur_phase : bool = False, blur_px : bool = True,
                       blur_size = None, ref_Fourier_normed : bool = False, regularize_sigma : float = 2.0
                       ) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Aligns an input image "im" to a reference image "ref". Uses the shift maximizing phase-correlation,
    but only the max integer-level pixel shift. Operates on one plane at a time -- you then align each plane to one
    another by adding a constant offset to all the frames discerned from the references.

    TODO: Maybe add subpixel shifts?

    INPUTS
    ------
    ref : numpy.ndarray

        2-dimensional numpy array REFERENCE image. May be a real frame or a constructed average/aligned frame.

    im: numpy.ndarray

        2-dimension numpy array to be aligned to ref.

    shift_only (optional) : bool

        Returns only the shift, does not return an array

    subpx (optional) : bool

        Align in the Fourier domain, permitting subpixel shifts. NOT YET IMPLEMENTED.

    phase_corr (optional) : bool

        The phase correlation plot itself

    blur_phase (optional) : bool

        Blur the phase correlation before taking the max (useful for noisy images)

    blur_px (optional) : bool

        Blurs correlation in the pixel domain, not in the Fourier domain

    blur_size (optional) : float

        Size of the phase correlation blur applied (in pixels)

    ref_Fourier_normed (optional) : bool

        Whether the input reference image has already been Fourier transformed.


    RETURNS
    -------------

    (shifted, (dy, dx)) if shift_only == False (default)
    (dy,dx) if shift_only == True
    (phase_correlation, (dy,dx)) if phase_corr == True and shift_only == False

    shifted : np.ndarray

        input image "im" shifted using numpy's roll method. Is not returned if shift_only is True.

    (dy,dx) : (int, int)

        pixel shifts in y direction and x direction respectively
    
    """

    if subpx:
        warnings.warn("Subpixel alignment not yet implemented. Ignoring"
        " this argument. Sorry!")

    # Take the Fourier transform of the reference image and normalize it.
    if not ref_Fourier_normed:
        ref = fft.fft2(ref)
        ref /= np.abs(ref)

    # Take the normalized complex conjugate of the Fourier transform of im
    im_fft = fft.fft2(im)
    im_fft = np.conjugate(im_fft)
    im_fft /= np.abs(im_fft)

    phase_product = ref*im_fft

    if blur_phase: # for when the phasecorr map is messy and you get a surprisingly large alignment shift
        if blur_size is None:
            blur_size = float(min(phase_product.shape)/30.0) # maybe there's a more principled way?
        
        xx, yy = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
        gauss = np.exp(-(xx/blur_size)**2)*np.exp(-(yy/blur_size))
        gauss /= np.sum(gauss)
        phase_product *= gauss
        #phase_product = scipy.ndimage.gaussian_filter(phase_product, sigma=blur_size, mode='wrap')

    pcorr = np.abs(fft.ifft2(phase_product))

    if blur_px:
        if blur_size is None:
            blur_size = float(min(phase_product.shape)/30.0)
        
        pcorr = scipy.ndimage.gaussian_filter(pcorr, sigma=blur_size, mode='wrap')

    offset = np.argmax(pcorr) # the index at which the inverse FFT of the product is maximum

    (dy,dx) = np.unravel_index(offset, ref.shape)

    (dy,dx) = (int(dy), int(dx))
    # various return types
    if shift_only:
        return (dy,dx)

    if phase_corr:
        return (pcorr,(dy,dx))

    # default
    return (np.roll(im, (dy,dx), axis=(0,1)), (dy,dx))

def align_references(reference_frames : list[np.ndarray], phase_blur : float = 10, regularize_sigma : float = 2.0, ignore_first : bool = True) -> list[tuple[int,int]]:
    """
    Takes a list of reference frames, aligns each to its adjacent planes. Returns a list of tuples
    corresponding to how each slice's frames should be shifted.

    Arguments
    ---------

    reference_frames : list[np.ndarray]

        List of images that are the reference for each 

    phase_blur : float (optional)

        How much to blur the phase correlation. The larger the number, the lower-pass the frequency content used for alignment.

    regularize_sigma : float (optional)

        How much to regularize the alignment across planes. Higher values mean stronger trust of the phase correlation, lower
        values mean to regress towards a shift of (0,0)

    ignore_first : bool (optional)

        Whether to ignore the first plane. Generally if you use the piezo-z you should do that.

    Returns
    -------

    shift_tuples : list[tuple[int,int]]

        One tuple for each frame with the y shift and x shift of the reference frame to align them all in z
    """

    ffts = np.fft.fft2(reference_frames)
    if ignore_first:
        ffts = np.fft.fft2([reference_frames[z] for z in range(1,len(reference_frames))])
    ffts /= np.abs(ffts)

    one_slice_pcorr = ffts * np.roll(np.conjugate(ffts),1, axis=0) # correlation offset by one slice
    relative_offsets = (np.array([ # list of tuples relative to one another. Element 0 = offset between 0 and 1. Element 1 = offset between 1 and 2
        np.unravel_index(
            np.argmax(
                scipy.ndimage.gaussian_filter(
                    np.abs(np.fft.ifft2(one_slice_pcorr[z])),
                    sigma=phase_blur,
                    mode='wrap'
                )
            ),
            ffts[0].shape
        )
        for z in range(ffts.shape[0])
    ]) + (tuple(t//2 for t in ffts[0].shape))) % (ffts[0].shape) - (tuple(t//2 for t in ffts[0].shape)) # zero it

    relative_offsets -= np.mean(relative_offsets,axis=0).astype(int) # minimize the total motion

    relative_offsets = relative_offsets % (ffts[0].shape)# unzero
    if ignore_first:
       return [(0,0)] + [tuple(offset) for offset in relative_offsets] 
    return [tuple(offset) for offset in relative_offsets]


def register_frames(siffio : SiffIO, frames : list[int], **kwargs)->tuple[dict, np.ndarray, np.ndarray]:
    """
    Registers the frames described by the list of indices in the input
    argument frames. This is for a single slice. To register multiple slices
    to one another, I register each slice independently and then align their
    reference frames (and use that to add a constant offset to each).

    Returns a tuple containing the registration dictionary, an array 
    of the distance between successive frames, and the reference image
    used for alignment. 

    Accepts kwargs for alignment as described below:

    INPUTS
    ------

    frames : list[int]

        The indices of the frames to be aligned.

    ref_method (optional) : str

        The method used to compute the reference frame. Options are
        detailed in the docstring for registration.build_reference_image.
        Currently available:

            'average'
            'suite2p'

    nimg_init (optional) : int

        Only used if ref_method is suite2p. Describes the number of frames
        from which the reference image may be constructed. Fewer is faster.

    seed_ref_count (optional) : int

        Only used if ref_method is suite2p. Describes the number of frames
        averaged to construct the reference image. Fewer is a sharper image.
        More is a more robust image.

    n_ref_iters (optional) : int

        Number of times to iteratively align. So it computes a reference image,
        aligns each frame to that reference image, but then can go back and compute
        a new reference image from aligned frames. n_ref_iters determines how many
        times to repeat this process. Default is 2: one naive and one with an aligned
        registration.

    subpx (optional) : bool

        Utilizes subpixel alignment in the Fourier phase correlation. NOT
        YET IMPLEMENTED!

    discard_bins (optional) : int

        Time bin above which photons should be discarded. Useful for known noise sources
        and known fluorophore IDs.
    
    tqdm (optional) : tqdm.tqdm

        For prettier output formatting in a notebook or script. Pass the tqdm.tqdm object.

    regularize_sigma (optional) : float

        Strength of the initial values from phase correlation vs. coupling to adjacent frames.
        Higher values mean stronger trust of the phase correlation.

    RETURNS
    -------

    (rdict, distances, ref_image)

        rdict : dict
            
            registration dict that can be passed to other siffpy functions

        distances : 1-d np.ndarray

            The distances between each successive frame during alignment.
            Useful for diagnosing the quality of the registration. If this
            is changing a lot, there's probably a bad alignment.

        ref_image : 2d np.ndarray

            The reference image used for the alignment.
    """

    if 'discard_bins' in kwargs:
        discard_bins = None
        if isinstance(kwargs['discard_bins'], int):
            discard_bins = kwargs['discard_bins']
    frames_np = siffio.get_frames(frames = frames, flim = False)
    use_tqdm = False
    if 'tqdm' in kwargs:
        import tqdm
        if isinstance(kwargs['tqdm'], tqdm.tqdm):
            use_tqdm = True
            pbar : tqdm.tqdm = kwargs['tqdm']
    
    regularize_sigma = 2.0
    if 'regularize_sigma' in kwargs:
        if isinstance(kwargs['regularize_sigma'], float):
            regularize_sigma = kwargs['regularize_sigma']

    import time
    t = time.time()
    ref_im = build_reference_image(siffio, frames, **kwargs)
    if use_tqdm:
        pbar.set_description(f"Ref image (1): {time.time() - t} seconds")

    # maybe there's a faster way to do this in one pass in numpy
    # I'll revisit it if registration starts to eat a lot of memory
    # or go super slow
    t = time.time()

    # Faster to just transform ref_im once
    ref_im_NORMED = fft.fft2(ref_im)
    ref_im_NORMED /= np.abs(ref_im_NORMED)
    if use_tqdm:    
        pbar.set_postfix({"Alignment iteration" : 1})
    
    rolls = [align_to_reference(ref_im_NORMED, frame, shift_only = True, ref_Fourier_normed=True) for frame in frames_np]
    
    if regularize_sigma > 0:
        # adjacent timepoints should be near each other.
        rolls = regularize_adjacent_tuples(rolls, ref_im.shape[0], ref_im.shape[1], sigma = regularize_sigma)
        
    reg_dict = {frames[n] : rolls[n] for n in range(len(frames))}
    
    n_ref_iters = 2
    if 'n_ref_iters' in kwargs:
        if isinstance(kwargs['n_ref_iters'], int):
            n_ref_iters = kwargs['n_ref_iters']

    # Repeat the same process -- build a reference, re-align
    for ref_iter in range(n_ref_iters - 1):
        if use_tqdm:
            pbar.set_postfix({"Alignment iteration" : ref_iter})
        t = time.time()
        ref_im = build_reference_image(siffio, frames, registration_dict = reg_dict, **kwargs)

        t = time.time()
        ref_im_NORMED = fft.fft2(ref_im)
        ref_im_NORMED /= np.abs(ref_im_NORMED)
        rolls = [align_to_reference(ref_im_NORMED, frame, shift_only = True, ref_Fourier_normed=True) for frame in frames_np]
        
        if regularize_sigma > 0:
            # adjacent timepoints should be near each other.
            rolls = regularize_adjacent_tuples(rolls, ref_im.shape[0], ref_im.shape[1], sigma = regularize_sigma)

        reg_dict = {frames[n] : rolls[n] for n in range(len(frames))}

    ysize,xsize = frames_np[0].shape

    # rebuild the reference images for storage.
    ref_im = build_reference_image(siffio, frames, registration_dict = reg_dict, **kwargs)

    roll_d_array = np.array([roll_d(rolls[n], rolls[n+1], ysize, xsize) for n in range(len(frames)-1)])

    return (reg_dict, roll_d_array, ref_im)

def regularize_adjacent_tuples(tuples : list[tuple], ydim : int, xdim: int, sigma : float = 2.0) -> list[tuple]:
    """
    Take a list of tuples, pretend adjacent ones are coupled by springs and to their original values.
    Find the minimum energy configuration. Sigma is the ratio of ORIGINAL to COUPLING.
    High sigma is more like the initial values. Low sigma pushes them all to identical values
    """
    s = (sigma**2)*np.ones(len(tuples))
    off = -1*np.ones(len(tuples)-1)
    trans_mat = diags(2+s) + diags(off,-1) + diags(off,1) # the forcing function is trans_mat * tuples - sigma**2 * tuples_init

    yz = spsolve(trans_mat,(sigma**2)*np.array([circ_d(roll_point[0], 0, ydim) for roll_point in tuples]))
    xz = spsolve(trans_mat,(sigma**2)*np.array([circ_d(roll_point[1], 0, xdim) for roll_point in tuples]))
    
    return [
        (
            int(re_circ(yz[k], ydim)),
            int(re_circ(xz[k], xdim))
        )
        for k in range(len(yz))
    ]

def regularize_all_tuples(tuples : list[tuple], ydim : int, xdim: int, sigma : float = 2.0) -> list[tuple]:
    """
    Take a list of tuples, pretend they're ALL coupled by springs and to their original values.
    Find the minimum energy configuration. Sigma is the ratio of ORIGINAL to COUPLING.
    High sigma is more like the initial values. Low sigma pushes them all to identical values
    """

    # Warn the user if they're applying a dangerous elasticity ratio. But don't prevent it.
    if np.abs(np.sqrt(np.abs(3-len(tuples))) - sigma) < 0.5:
        warnings.warn(
                f"""USING SINGULAR VALUE FOR ELASTIC REGULARIZATION PARAMETER
                ( SIGMA IS WITHIN 0.5 OF SQRT(NUM_SLICES - 3) ).
                You used sigma = {sigma} for {len(tuples)} slices. 
                EXPECT WEIRD RESULTS."
                """
            )

    s = (3+sigma**2)*np.diag(np.ones(len(tuples)))
    trans_mat = s - 1

    yz = solve(trans_mat,(sigma**2)*np.array([circ_d(roll_point[0], 0, ydim) for roll_point in tuples]))
    xz = solve(trans_mat,(sigma**2)*np.array([circ_d(roll_point[1], 0, xdim) for roll_point in tuples]))
    
    return [
        (
            int(re_circ(yz[k], ydim)),
            int(re_circ(xz[k], xdim))
        ) 
        for k in range(len(yz))
    ]
