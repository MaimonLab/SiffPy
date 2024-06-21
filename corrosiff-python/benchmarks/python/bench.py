import numpy as np
import timeit
from typing import Callable

# The files used to benchmark the Python implementations
files = [
    '/Users/stephen/Desktop/Data/imaging/2024-04/2024-04-17/21Dhh_GCaFLITS/Fly1/Flashes_1.siff',
    '/Users/stephen/Desktop/Data/imaging/2024-05/2024-05-27/R60D05_TqCaFLITS/Fly1/EBAgain_1.siff',
    '/Users/stephen/Desktop/Data/imaging/2024-05/2024-05-27/SS02255_greenCamui_alpha/Fly1/PB_1.siff',
]

def run_test_on_file(file : str, test : Callable, num_iters : int = 10)->np.ndarray:
    """
    Creates two siffreaders, one with the `corrosiff` backend
    and one with the `siffreadermodule` backend, and compares
    the execution times of a test function. Expects "test"
    to take a single argument: the siffreader of interest
    """

    setup = f"""import numpy as np
from siffpy import SiffReader
from __main__ import {test.__name__} as test;
rust_reader = SiffReader('{file}', backend='corrosiff')
cpp_reader = SiffReader('{file}', backend='siffreadermodule')
    """

    # Run the test on the rust reader
    rust_times = timeit.repeat(
        stmt = "test(rust_reader)",
        setup = setup,
        number = num_iters,
    )

    # Run the test on the cpp reader
    cpp_times = timeit.repeat(
        stmt = "test(cpp_reader)",
        setup = setup,
        number = num_iters,
    )

    # Return the times
    return np.array([rust_times,cpp_times])