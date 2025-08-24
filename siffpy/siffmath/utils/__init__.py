from typing import Any, Tuple, Union, TypeVar

import numpy as np
from scipy.ndimage import convolve
from siffpy.siffmath.utils.timeseries import Timeseries, TimeseriesUnits # noqa

T = TypeVar('T', bound=np.generic)

def local_correlation(
    frames : np.ndarray[Any, T],
    neighborhood_size : Union[int, Tuple[int]] = (1,3,3),
) -> np.ndarray[Any, float]:
    """
    Computes the correlation of each pixel with its neighbors,
    returning a large value if the pixel is well correlated with its neighbors,
    and a small value if it is not. Useful for creating a default map to draw
    the ROIs on. Unfortunately this can be kind of slow for large arrays because
    there's no batching.

    Arguments
    ---------
    - frames : np.ndarray

        3+D array of image frames, timepoints with the first axis being the "time"
        axis, and the remaining axes being spatial dimensions (probably should isolate
        a single color channel though or else this will be convolved across
        the color channels as well)

    - neighborhood_size : int
    
        The size of the neighborhood to consider for the correlation. Must be an odd
        number, and should be small enough to not include too many pixels in the
        neighborhood.

    Returns
    -------
    - np.ndarray

        3+D array of the same shape as `frames`, with the first axis being the "time"
        axis, and the remaining axes being spatial dimensions. The values are the
        correlation coefficients of each pixel with its neighbors.

    Raises
    ------
    - ValueError

        If the neighborhood size is not an odd number in any dimension.

    Example
    -------
    ```python
    >>> frames = np.random.rand(10, 100, 100)  # 10 timepoints, 100x100 spatial dimensions
    >>> local_corr = local_correlation(frames, neighborhood_size=5)
    >>> print(local_corr.shape)
    ```
    """
    frames = frames.astype(np.float32, copy = True)
    mean = frames.mean(axis=0, keepdims=True)
    std = frames.std(axis=0, keepdims=True)
    np.divide((frames - mean) , std, out=frames)

    if isinstance(neighborhood_size, int):
        if neighborhood_size % 2 == 0:
            raise ValueError("Neighborhood size must be an odd number")
    
        kernel = np.ones((neighborhood_size,)* (frames.ndim - 1))
        # exclude the center pixel from the kernel
        kernel[(neighborhood_size//2,) * (frames.ndim - 1)] = 0
    
    elif isinstance(neighborhood_size, tuple):
        if len(neighborhood_size) != frames.ndim - 1:
            raise ValueError(
                f"Neighborhood size must be a single integer or a tuple of length {frames.ndim - 1}, but got {neighborhood_size}"
            )
        if any(s % 2 == 0 for s in neighborhood_size):
            raise ValueError("All dimensions of neighborhood size must be odd numbers")
        kernel = np.ones(neighborhood_size)
        # exclude the center pixel from the kernel
        kernel[tuple(s//2 for s in neighborhood_size)] = 0
    else:
        raise TypeError("Neighborhood size must be an int or a tuple of ints")
    
    kernel /= kernel.sum()

    conv = convolve(
        frames,
        kernel[np.newaxis, ...],  # Add a new axis for the time dimension
        mode='nearest',
        cval=0.0,
    )

    return (frames * conv).mean(axis=0)