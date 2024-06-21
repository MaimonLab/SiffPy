import pytest
import corrosiffpy
import siffreadermodule
import numpy as np

from siffpy.core.flim import FLIMParams, Exp, Irf

def test_read_histogram(siffreaders):
    corrosiff_sr, siffc_sr = siffreaders

    assert (
        corrosiff_sr.get_histogram()
        == siffc_sr.get_histogram()[:629]
    ).all()

    assert(
        corrosiff_sr.get_histogram_by_frames().sum(axis =0)
        == siffc_sr.get_histogram()[:629]
    ).all()

def test_read_flim_frames(siffreaders):
    corrosiff_sr, siffc_sr = siffreaders

    test_params = FLIMParams(
        Exp(tau = 0.5, frac = 0.5, units = 'nanoseconds'),
        Exp(tau = 2.5, frac = 0.5, units = 'nanoseconds'),
        Irf(offset = 1.1, sigma = 0.2, units = 'nanoseconds'),
    )

    with test_params.as_units('countbins'):
        assert np.allclose(
            corrosiff_sr.flim_map(test_params, registration=None)[0],
            siffc_sr.flim_map(test_params, registration=None)[0],
            equal_nan = True
        )

    dummy_reg = {
        k : (
        int(np.random.uniform(low = -128, high = 128)) % 128,
        int(np.random.uniform(low = -128, high = 128)) % 128
        ) for k in range(10000)
    }
    framelist = list(range(10000))

    with test_params.as_units('countbins'):
        assert not np.allclose(
            corrosiff_sr.flim_map(params = test_params, frames = framelist, registration=None)[0],
            siffc_sr.flim_map(params = test_params, frames = framelist, registration=dummy_reg)[0],
            equal_nan = True
        )

        assert np.allclose(
            corrosiff_sr.flim_map(params = test_params, frames = framelist, registration=dummy_reg)[0],
            siffc_sr.flim_map(params = test_params, frames = framelist, registration=dummy_reg)[0],
            equal_nan = True
        )

# def test_sum_2d_mask(siffreaders):
#     corrosiff_sr, siffc_sr = siffreaders

#     roi = np.random.rand(*corrosiff_sr.frame_shape()) > 0.3
#     assert (
#         corrosiff_sr.sum_roi(roi, registration=None)
#         == siffc_sr.sum_roi(roi, registration=None)
#     ).all()

#     dummy_reg = {
#         k : (
#         int(np.random.uniform(low = -128, high = 128)) % 128,
#         int(np.random.uniform(low = -128, high = 128)) % 128
#         ) for k in range(10000)
#     }
#     framelist = list(range(10000))

#     assert not (
#         corrosiff_sr.sum_roi(roi, frames = framelist, registration=None)
#         == siffc_sr.sum_roi(roi, frames = framelist, registration=dummy_reg)
#     ).all()

#     assert (
#         corrosiff_sr.sum_roi(roi, frames = framelist, registration=dummy_reg)
#         == siffc_sr.sum_roi(roi, frames = framelist, registration=dummy_reg)
#     ).all()