"""
A submodule containing methods for performing color channel
correction analyses to estimate bleedthrough
"""
from typing import Tuple, Any

import numpy as np

from ..utils.types import ImageArray

def correct_bleedthrough_linear(
        x : np.ndarray,
        y : np.ndarray,
        x_to_y : float = 0.0,
        y_to_x : float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Corrects bleedthrough between two color channels assuming the relationships:

    $ y_true(t) = y(t) - x_to_y * x_true(t) $

    and

    $ x_true(t) = x(t) - y_to_x * y_true(t) $

    ## Arguments

    - `x` :
        An array containing the measurements of one series

    - `y` :
        An array containing the measurements of a second series

    - `x_to_y` :
        The estimated proportional factor of _true_ `x` value that contaminates
        the signal of `y`

    - `y_to_x` :
        The estimated proportional factor of _true_ `y` value that contaminates
        the signal of `x`

    ## Returns

    (x_corrected, y_corrected)
    
    """
    return (
        np.maximum((x - y_to_x*y)/(1-(x_to_y*y_to_x)), 0),
        np.maximum((y - x_to_y*x)/(1-(x_to_y*y_to_x)), 0),
    )

def linear_fit(*image_channels : Tuple[ImageArray])->np.ndarray[Any, np.float64]:
    """
    Estimates a linear fit from each channel to the others.

    Arguments
    ---------

    image_channels : Tuple[ImageArray]
        Each element of the tuple is an image array,
        either a timeseries for the channel or pixelwise
        intensity values (in time series form).

    Returns
    -------

    bleedthrough_matrix : np.ndarray
        A matrix of shape (num_channels, num_channels)
        where the element bleedthrough_matrix[i,j] is the
        amount of channel j signal that bleeds into channel i.
    """

    if not all([image_channel.shape == image_channels[0].shape
                for image_channel in image_channels]):
        raise ValueError("All image channels must have the same shape")
    raise NotImplementedError("`linear_fit` not yet implemented")