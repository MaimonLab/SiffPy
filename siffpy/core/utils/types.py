from typing import Union, Any
from pathlib import Path

import numpy as np

PathLike = Union[str, Path]

ImageArray = np.ndarray[Any, np.dtype[np.uintc]]
BoolMaskArray = np.ndarray[Any, np.dtype[np.bool_]]