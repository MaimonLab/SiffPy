from typing import Callable, Any
import numpy as np


PDF_Function = Callable[[np.ndarray[Any, np.float64]], np.ndarray[Any, np.float64]]
Objective_Function = Callable[[np.ndarray[Any, np.float64]], np.ndarray[Any, np.float64]]