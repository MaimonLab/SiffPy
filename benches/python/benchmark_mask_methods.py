"""
Contains benchmarking for the opening time of files.
"""

from siffpy import SiffReader
from local_consts import large_path

import numpy as np

def path_to_setup_str(path : str):
    """ Returns a string that can be used in a
    timeit setup argument to open a file """
    #return f"\nfrom siffpy import SiffReader\nsr = SiffReader('{path}')"


def test_2d_sum_masks_iterative(sr : 'SiffReader', masks):
    masks_out = np.array([sr.sum_mask(mask, framewise = True, registration_dict={}) for mask in masks])

def test_2d_sum_masks_together(sr : 'SiffReader', masks):
    masks_out = sr.sum_masks(masks, framewise = True, registration_dict={})

def test_3d_sum_masks_iterate(sr : 'SiffReader', masks):
    masks_out = np.array([sr.sum_mask(mask) for mask in masks])
    pass

def test_3d_sum_masks_together(sr : 'SiffReader', masks):
    masks_out = sr.sum_masks(masks)
    pass

if __name__ == '__main__':
    import timeit
    
    sr = SiffReader(large_path)
    flat_masks = [np.ones(sr.im_params.shape).astype(bool) for _ in range(10)]
    flat_arr = np.array(flat_masks)
    three_d_masks = [np.ones(sr.im_params.single_channel_volume).astype(bool) for _ in range(10)]
    three_d_arr = np.array(three_d_masks)

    n = 5
    print(
        "Sum 10 one-plane masks iteratively, combine to array:\n",
        timeit.timeit(
            "test(sr, masks)",
            setup=("from __main__ import test_2d_sum_masks_iterative as test"
            + "\nfrom __main__ import flat_masks as masks"
            + "\nfrom __main__ import sr"),
            #+ path_to_setup_str(large_path),
            number = n,
        )/n , "sec per iter"
    )

    print(
        "Sum 10 one-plane masks together:\n",
        timeit.timeit(
            "test(sr, masks)",
            setup=("from __main__ import test_2d_sum_masks_together as test"
            + "\nfrom __main__ import flat_arr as masks"
            + "\nfrom __main__ import sr"),
            number = n,
        )/n , "sec per iter"
    )

    # print(
    #     "Sum 10 multi-plane masks iteratively, combine to array:\n",
    #     timeit.timeit(
    #         "test(sr, masks)",
    #         setup="from __main__ import test_3d_sum_masks_iterate as test"
    #         + path_to_setup_str(large_path),
    #         number = 10,
    #     )/10 , "sec per iter"
    # )

    # print(
    #     "Sum 10 multi-plane masks together:\n",
    #     timeit.timeit(
    #         "test(sr, masks)",
    #         setup="from __main__ import test_3d_sum_masks_together as test"
    #         + path_to_setup_str(large_path),
    #         number = 10,
    #     )/10 , "sec per iter"
    # )