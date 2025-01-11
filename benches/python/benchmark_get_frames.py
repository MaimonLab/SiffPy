"""
Contains benchmarking for the opening time of files.
"""

import siffpy

from local_consts import small_path, large_path

def test_read_small(sr):
    sr.get_frames(frames = list(range(40)), registration_dict = {})

def test_read_large(sr):
    sr.get_frames(frames = list(range(50000)), registration_dict = {})

if __name__ == '__main__':
    import timeit
    sr = siffpy.SiffReader(small_path, backend = 'siffreadermodule')
    print(
        "Get 40 low photon count frames (C++):\n",
        timeit.timeit(
            "test(sr)",
            setup="from __main__ import test_read_small as test\n"
            + "from __main__ import sr\n",
            number = 100,
        )/100 * 1000 , "msec per iter"
    )
    src = siffpy.SiffReader(small_path, backend = 'corrosiff')
    print(
        "Get 40 low photon count frames (Rust):\n",
        timeit.timeit(
            "test(sr)",
            setup="from __main__ import test_read_small as test\n"
            + "from __main__ import src as sr",
            number = 100,
        )/100 * 1000 , "msec per iter"
    )
    sr = siffpy.SiffReader(large_path, backend = 'siffreadermodule')
    print(
        "Get 50000 many-photon frames (C++):\n",
        timeit.timeit(
            "test(sr)",
            setup="from __main__ import test_read_large as test"
            + "\nfrom __main__ import sr",
            number = 10,
        )/10 , "sec per iter"
    )

    src = siffpy.SiffReader(large_path, backend = 'corrosiff')
    print(
        "Get 50000 many-photon frames (Rust):\n",
        timeit.timeit(
            "test(sr)",
            setup="from __main__ import test_read_large as test"
            + "\nfrom __main__ import src as sr",
            number = 10,
        )/10 , "sec per iter"
    )