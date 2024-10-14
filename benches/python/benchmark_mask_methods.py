"""
Contains benchmarking for the opening time of files.
"""

from siffpy import SiffReader
from local_consts import large_path

import numpy as np

COMPARE_TO_APPLYING_MASK_TO_WHOLE_FRAMES = False

def test_2d_sum_masks_iterative(sr : 'SiffReader', masks):
    """ Iteratively uses `sum_mask` then concatenates into an array"""
    np.array([sr.sum_mask(mask, framewise = True, registration_dict={}) for mask in masks])

def test_2d_sum_masks_together(sr : 'SiffReader', masks):
    """ Uses the `siffpy` sum_masks method"""
    sr.sum_masks(masks, framewise = True, registration_dict={})

def test_apply_mask_to_whole_frames_direct(sr : 'SiffReader', masks):
    """
    Reads all frames and then applies the masks individually (then
    concatenates into an array)
    """
    frames = (
        sr.get_frames(frames = sr.im_params.flatten_by_timepoints())
        .reshape((-1,*sr.im_params.volume_one_color)).squeeze()
    )
    np.array([
        frames[..., mask].sum()
        for mask in masks
    ])

def test_3d_sum_masks_iterate(sr : 'SiffReader', masks):
    np.array([sr.sum_mask(mask) for mask in masks])

def test_3d_sum_masks_together(sr : 'SiffReader', masks):
    sr.sum_masks(masks)

if __name__ == '__main__':
    import timeit
    
    sr = SiffReader(large_path)
    flat_masks = [np.ones(sr.im_params.shape).astype(bool) for _ in range(10)]
    flat_arr = np.array(flat_masks)
    three_d_masks = [np.ones(sr.im_params.single_channel_volume).astype(bool) for _ in range(10)]
    three_d_arr = np.array(three_d_masks)

    n = 5

    if COMPARE_TO_APPLYING_MASK_TO_WHOLE_FRAMES:
        print(
            "Apply one-plane masks after loading all frames:\n",
            timeit.timeit(
                "test(sr, masks)",
                setup=("from __main__ import test_apply_mask_to_whole_frames_direct as test"
                + "\nfrom __main__ import flat_arr as masks"
                + "\nfrom __main__ import sr"),
                number = n,
            )/n , "sec per iter ({n} iters)"
        )

    print(
        "Sum 10 one-plane masks iteratively, combine to array:\n",
        timeit.timeit(
            "test(sr, masks)",
            setup=("from __main__ import test_2d_sum_masks_iterative as test"
            + "\nfrom __main__ import flat_masks as masks"
            + "\nfrom __main__ import sr"),
            #+ path_to_setup_str(large_path),
            number = n,
        )/n , f"sec per iter ({n} iters)"
    )

    print(
        "Sum 10 one-plane masks together:\n",
        timeit.timeit(
            "test(sr, masks)",
            setup=("from __main__ import test_2d_sum_masks_together as test"
            + "\nfrom __main__ import flat_arr as masks"
            + "\nfrom __main__ import sr"),
            number = n,
        )/n , "sec per iter ({n} iters)"
    )

    print(
        "Sum 10 multi-plane masks iteratively, combine to array:\n",
        timeit.timeit(
            "test(sr, masks)",
            setup="from __main__ import test_3d_sum_masks_iterate as test"
            + "\nfrom __main__ import three_d_masks as masks"
            + "\nfrom __main__ import sr",
            number = 10,
        )/10 , "sec per iter"
    )

    print(
        "Sum 10 multi-plane masks together:\n",
        timeit.timeit(
            "test(sr, masks)",
            setup="from __main__ import test_3d_sum_masks_together as test"
            + "\nfrom __main__ import three_d_arr as masks"
            + "\nfrom __main__ import sr",
            number = 10,
        )/10 , "sec per iter"
    )