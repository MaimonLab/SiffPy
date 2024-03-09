from typing import Callable, Any
import numpy as np


PDF_Function = Callable[[np.ndarray[Any, np.float_]], np.ndarray[Any, np.float_]]
Objective_Function = Callable[[np.ndarray[Any, np.float_]], np.ndarray[Any, np.float_]]