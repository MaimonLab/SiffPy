import numpy as np
from numpy import fft
import siffreader
import warnings, random, scipy

def build_reference_image(frames : list[int], ref_method : str = 'suite2p', **kwargs) -> np.ndarray:
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
        for siffpy.siffutils.registration.suite2p_reference to learn more.

    RETURNS
    -------
    np.ndarray:

        A reference image built from the input frames
    """

    if ref_method == 'average':
        return siffreader.pool_frames(pool_lists=[frames], flim = False)

    if ref_method == 'suite2p':
        return suite2p_reference(frames, **kwargs)

    raise TypeError("Method passed for constructing average not valid")

def suite2p_reference(frames : list[int], **kwargs) -> np.ndarray:
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

    nimg_init = min(len(frames)/10, 200)
    if 'nimg_init' in kwargs:
        nimg_init = kwargs['nimg_init']
        if not isinstance(nimg_init, int):
            warnings.warn("Suite2p alignment arg 'nimg_init' is not of type int. "
                          "Using default value instead.")
            nimg_init = min(len(frames/10),200)
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

    init_frames = np.array(
                    siffreader.get_frames(frames = init_frames_idx, flim = False, registration = registration_dict)
                  )
    
    # find the few frame most correlated with the mean of these

    mean_img = np.nanmean(init_frames,axis=0)
    mean_subbed  = (init_frames - mean_img)**2
    err_val = np.sum(mean_subbed,axis=(1,2))

    # how many of the most correlated frames to take
    seed_ref_count = 20
    if 'seed_ref_count' in kwargs:
        seed_ref_count = kwargs['seed_ref_count']
        if not isinstance(seed_ref_count, int):
            warnings.warn("Suite2p alignment arg 'seed_ref_count' is not of type int. "
                          "Using 20 instead.")
            seed_ref_count = 20
        if seed_ref_count > nimg_init:
            warnings.warn("Suite2p alignment arg 'seed_ref_count' is greater than number "
                          f"of frames being aligned. Defaulting to 20.")
            seed_ref_count = 20
    seed_idx = np.argpartition(err_val, seed_ref_count)[:seed_ref_count]
    return np.squeeze(np.mean(init_frames[seed_idx,:,:],axis=0))

def align_to_reference(ref : np.ndarray, im : np.ndarray, shift_only : bool = False, 
                       subpx : bool = False, phase_corr : bool = False, blur : bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Aligns an input image "im" to a reference image "ref". Uses the shift maximizing phase-correlation,
    but only the max integer-level pixel shift. TODO: Maybe add subpixel shifts?

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

    blur (optional) : bool

        Blur the phase correlation before taking the max (useful for noisy images)

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
    ref = fft.fft2(ref)
    ref /= np.abs(ref)

    # Take the normalized complex conjugate of the Fourier transform of im
    im2 = fft.fft2(im)
    im2 = np.conjugate(im2)
    im2 /= np.abs(im2)

    pcorr = np.abs(fft.ifft2(ref*im2))

    if blur: # for when the phasecorr map is messy and you get a surprisingly large alignment shift
        import scipy.ndimage
        blur_size = float(min(pcorr.shape)/100.0)
        pcorr = -scipy.ndimage.gaussian_laplace(pcorr, sigma=blur_size,mode='wrap')
    
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

def register_frames(frames : list[int], **kwargs)->tuple[dict, np.ndarray, np.ndarray]:
    """
    Registers the frames described by the list of indices in the input
    argument frames.

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

    frames_np = siffreader.get_frames(frames = frames, flim = False)
    import time
    t = time.time()
    ref_im = build_reference_image(frames, **kwargs)
    print(f"Ref image (1): {time.time() - t} seconds")
    # maybe there's a faster way to do this in one pass in numpy
    # I'll revisit it if registration starts to eat a lot of memory
    # or go super slow
    t = time.time()
    rolls = [align_to_reference(ref_im, frame, shift_only = True) for frame in frames_np]
    print(f"Align images (1): {time.time() - t} seconds")

    reg_dict = {frames[n] : rolls[n] for n in range(len(frames))}
    
    n_ref_iters = 2
    if 'n_ref_iters' in kwargs:
        if isinstance(kwargs['n_ref_iters'], int):
            n_ref_iters = kwargs['n_ref_iters']

    for ref_iter in range(n_ref_iters - 1):
        t = time.time()
        ref_im = build_reference_image(frames, registration_dict = reg_dict, **kwargs)
        print(f"Ref image ({ref_iter + 1}): {time.time() - t} seconds")

        t = time.time()
        rolls = [align_to_reference(ref_im, frame, shift_only = True) for frame in frames_np]
        print(f"Align images ({ref_iter + 1}): {time.time() - t} seconds")

        reg_dict = {frames[n] : rolls[n] for n in range(len(frames))}

    ysize,xsize = frames_np[0].shape

    roll_d_array = np.array([roll_d(rolls[n], rolls[n+1], ysize, xsize) for n in range(len(frames)-1)])

    return (reg_dict, roll_d_array, ref_im)

def registration_cleanup(registration_tuple : tuple, frame_idxs : list[int])-> tuple[dict, np.ndarray, np.ndarray]:
    """
    Take in a tuple from register_frames above and try to
    clean it up a bit by identifying badly behaving frames
    and seeing if we can't find a slightly better alignment
    for them by blurring the phase correlation map.

    TODO: CLEANUP IN B
    """

    estimated_shifts = registration_tuple[1]
    velocity = np.diff(estimated_shifts) # we expect this to be near constant

    # find where the velocity is varying a lot
    jittery_frames_idx = np.where((velocity-np.mean(velocity))/np.std(velocity) > 2.5)[0]
    if len(jittery_frames_idx):
        warnings.warn("Some frames seem not to have aligned well. Trying again with"
                      f" a blurred phase correlation. Fixing {len(jittery_frames_idx)} frames"
        )
    
    print(registration_tuple[1][jittery_frames_idx])
    # the above is in term of the frame indices within this FOV, not the
    # actual frame number I need for the siffreader
    absolute_idxs = [frame_idxs[frame] for frame in jittery_frames_idx]

    naughty_frames = siffreader.get_frames(frames=absolute_idxs, flim=False)
    new_shifts = [align_to_reference(registration_tuple[2], nframe, blur = True, shift_only = True) for nframe in naughty_frames]

    # remap the registration dict
    for n in range(len(absolute_idxs)):
        registration_tuple[0][absolute_idxs[n]] = new_shifts[n]
    
    #TODO CLEAN UP REGISTRATION_TUPLE[1] WITH NEW INFO
    return registration_tuple

def circ_d(x : float, y : float, rollover : float)->float:
    """Wrapped-around distance between x and y"""
    return min((x-y) % rollover, (y-(x-rollover)) % rollover)

def roll_d(roll1 : tuple[float, float], roll2: tuple[float,float], rollover_y: float, rollover_x : float)->float:
    """ Distance between two rollovers """
    d_y = circ_d(roll1[0],roll2[0],rollover_y)
    d_x = circ_d(roll1[1],roll2[1],rollover_x)
    return np.sqrt(d_x**2 + d_y**2)