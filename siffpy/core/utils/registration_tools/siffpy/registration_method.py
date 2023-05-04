import logging

import numpy as np
import numpy.fft as fft
from numpy.linalg import solve
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from siffreadermodule import SiffIO
from siffpy.core.utils.circle_fcns import circ_d, re_circ, roll_d
from siffpy.core.utils.registration_tools.siffpy.alignment import (
    align_to_reference
)

def register_frames(
        siffio : SiffIO,
        reference_frame : np.ndarray,
        frames : list[int],
        registration_dict : dict = {},
        pbar = None,
        regularize_sigma : float = 2.0,
        **kwargs)->dict[int, tuple[int, int]]:
    """
    Registers the frames described by the list of indices in the input
    argument frames. This is for a single slice. To register multiple slices
    to one another, I register each slice independently and then align their
    reference frames (and use that to add a constant offset to each).

    Returns the registration dictionary

    Accepts kwargs for alignment as described below:

    INPUTS
    ------

    siffio : SiffIO

        The filereader which reads the frames to be aligned.

    reference_frame : np.ndarray

        A single reference image to which the frames will be aligned.

    frames : list[int]

        The indices of the frames to be aligned.

    pbar (optional) : tqdm.tqdm

        For prettier output formatting in a notebook or script. Pass the tqdm.tqdm object.

    regularize_sigma (optional) : float

        Strength of the initial values from phase correlation vs. coupling to adjacent frames.
        Higher values mean stronger trust of the phase correlation.

    RETURNS
    -------

    rdict : dict
            
        registration dict that can be passed to other siffpy functions
    """

    frames_np = siffio.get_frames(frames = frames, registration=registration_dict)
    use_tqdm = not (pbar is None)
    
    #if use_tqdm:
    #    pbar.set_description(f"\nRef image (1): {time.time() - t} seconds")

    # maybe there's a faster way to do this in one pass in numpy
    # I'll revisit it if registration starts to eat a lot of memory
    # or go super slow

    # Faster to just transform ref_im once
    ref_im_NORMED = fft.fft2(reference_frame)
    ref_im_NORMED /= np.abs(ref_im_NORMED)
    #if use_tqdm:    
    #    pbar.set_postfix({"Alignment iteration" : 1})
    
    rolls = [align_to_reference(ref_im_NORMED, frame, shift_only = True, ref_Fourier_normed=True) for frame in frames_np]
    
    if regularize_sigma > 0:
        # adjacent timepoints should be near each other.
        rolls = regularize_adjacent_tuples(
            rolls,
            reference_frame.shape[0],
            reference_frame.shape[1],
            sigma = regularize_sigma
        )
        
    return {frames[n] : rolls[n] for n in range(len(frames))}
    
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