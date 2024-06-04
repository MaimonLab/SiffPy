"""
Contains benchmarking for the opening time of files.
"""

import siffpy

from local_consts import small_path, large_path

def path_to_setup_str(path : str):
    """ Returns a string that can be used in a
    timeit setup argument to open a file """
    return f"\nfrom siffpy import SiffReader\nsr = SiffReader('{path}')"


def test_read_small(sr):
    sr.get_frames(frames = list(range(40)))

def test_read_large(sr):
    sr.get_frames(frames = list(range(50000)))

if __name__ == '__main__':
    import timeit
    sr = siffpy.SiffReader(small_path)
    print(
        "Get 40 low photon count frames:\n",
        timeit.timeit(
            "test(sr)",
            setup="from __main__ import test_read_small as test"
            + path_to_setup_str(small_path),
            number = 100,
        )/100 * 1000 , "msec per iter"
    )
    sr = siffpy.SiffReader(large_path)
    print(
        "Get 50000 many-photon frames:\n",
        timeit.timeit(
            "test(sr)",
            setup="from __main__ import test_read_large as test"
            + path_to_setup_str(large_path),
            number = 10,
        )/10 , "sec per iter"
    )