from typing import Union, Any
from pathlib import Path

import numpy as np

PathLike = Union[str, Path]

ImageArray = 'np.ndarray[Any, np.dtype[np.uintc]]'
BoolMaskArray = 'np.ndarray[Any, np.dtype[np.bool_]]'
FloatArray = 'np.ndarray[Any, np.dtype[np.float64]]'
ComplexArray = 'np.ndarray[Any, np.dtype[np.complex128]]'