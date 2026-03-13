from typing import Callable, Any
import numpy as np


PDF_Function = Callable[[np.ndarray[Any, np.dtype[np.floating]]], np.ndarray[Any, np.dtype[np.floating]]]
Objective_Function = Callable[[np.ndarray[Any, np.dtype[np.floating]]], np.ndarray[Any, np.dtype[np.floating]]]