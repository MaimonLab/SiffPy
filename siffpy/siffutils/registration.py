import numpy as np
from numpy import fft

def align_to_reference(ref : np.ndarray, im : np.ndarray, shift_only : bool = False) -> (np.ndarray, (int, int)):
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

    RETURNS
    -------------

    (shifted, (dy, dx))

    shifted : np.ndarray

        input image "im" shifted using numpy's roll method

    (dy,dx) : (int, int)

        pixel shifts in y direction and x direction respectively
    
    """

    # Take the Fourier transform of the reference image and normalize it.
    ref = fft.fft2(ref)
    ref /= np.abs(ref)

    # Take the normalized complex conjugate of the Fourier transform of im
    im2 = fft.fft2(im)
    im2 = np.conjugate(im2)
    im2 /= np.abs(im2)

    offset = np.argmax(np.abs(fft.ifft2(ref*im2))) # the index at which the inverse FFT of the product is maximum

    (dy,dx) = np.unravel_index(offset, ref.shape)

    if shift_only:
        return (dy,dx)

    return (np.roll(im, (dy,dx), axis=(0,1)), (dy,dx))