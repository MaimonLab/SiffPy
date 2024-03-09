"""
A submodule containing code for using arrival time data
to better correct for color channel bleedthrough across
imaging channels.
"""

from typing import Tuple, Any

import numpy as np

from .traces import FlimTrace

def linear_fit(*image_channels : Tuple[FlimTrace])->np.ndarray[Any, np.float_]:
    raise NotImplementedError()