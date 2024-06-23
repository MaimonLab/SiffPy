"""
A submodule containing methods for performing color channel
correction analyses to estimate bleedthrough
"""
from typing import Tuple, Any

import numpy as np

from ..utils.types import ImageArray

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
    raise NotImplementedError("linear_fit not yet implemented")