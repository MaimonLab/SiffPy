import logging

import numpy as np
import numpy.fft as fft
from numpy.linalg import solve
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from siffreadermodule import SiffIO
from siffpy.core.utils.circle_fcns import circ_d, re_circ, roll_d
from siffpy.core.utils.registration_tools.siffpy.alignment import (
    align_to_reference, build_reference_image
)

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
        logging.warning(
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