import corrosiffpy
import siffreadermodule
import numpy as np

def test_read_frames(siffreaders):
    corrosiff_sr : corrosiffpy.SiffIO = siffreaders[0]
    siffc_sr : siffreadermodule.SiffIO = siffreaders[1]

    assert (
        corrosiff_sr.get_frames(registration=None)
        == siffc_sr.get_frames(registration=None)
    ).all()

    dummy_reg = {
        k : (
        int(np.random.uniform(low = -128, high = 128)) % 128,
        int(np.random.uniform(low = -128, high = 128)) % 128
        ) for k in range(10000)
    }
    framelist = list(range(10000))

    assert not (
        corrosiff_sr.get_frames(frames = framelist, registration=None)
        == siffc_sr.get_frames(frames = framelist, registration=dummy_reg)
    ).all()

    assert (
        corrosiff_sr.get_frames(frames = framelist, registration=dummy_reg)
        == siffc_sr.get_frames(frames = framelist, registration=dummy_reg)
    ).all()

def test_sum_2d_mask(siffreaders):
    corrosiff_sr : corrosiffpy.SiffIO = siffreaders[0]
    siffc_sr : siffreadermodule.SiffIO = siffreaders[1]

    roi = np.random.rand(*corrosiff_sr.frame_shape()) > 0.3
    assert (
        corrosiff_sr.sum_roi(roi, registration=None)
        == siffc_sr.sum_roi(roi, registration=None)
    ).all()

    dummy_reg = {
        k : (
        int(np.random.uniform(low = -128, high = 128)) % 128,
        int(np.random.uniform(low = -128, high = 128)) % 128
        ) for k in range(10000)
    }
    framelist = list(range(10000))

    assert not (
        corrosiff_sr.sum_roi(roi, frames = framelist, registration=None)
        == siffc_sr.sum_roi(roi, frames = framelist, registration=dummy_reg)
    ).all()

    assert (
        corrosiff_sr.sum_roi(roi, frames = framelist, registration=dummy_reg)
        == siffc_sr.sum_roi(roi, frames = framelist, registration=dummy_reg)
    ).all()

def test_sum_3d_mask(siffreaders):
    corrosiff_sr, siffc_sr = siffreaders

    NUM_PLANES = 7

    rois = [np.random.rand(k, *corrosiff_sr.frame_shape()) > 0.3 for k in range(1,NUM_PLANES)]

    # Validate that they both cycle through the same way
    for k in range(1,NUM_PLANES):
        N_FRAMES = 10000 - (10000 % k)

        # C++ API is consistent
        assert (
            np.array([
                siffc_sr.sum_roi(rois[k-1][p], frames = list(range(p, N_FRAMES ,k)),registration=None)
                for p in range(k)
            ]).T.flatten()
            == siffc_sr.sum_roi(
                rois[k-1], frames = list(range(N_FRAMES)), registration=None
            ).flatten()
        ).all()

        # # Slicewise agrees
        assert (
           np.array([
                siffc_sr.sum_roi(rois[k-1][p], frames = list(range(p, N_FRAMES ,k)),registration=None)
                for p in range(k)
            ]).T.flatten()
            == np.array([
                corrosiff_sr.sum_roi(rois[k-1][p], frames = list(range(p, N_FRAMES ,k)),registration=None)
                for p in range(k)
            ]).T.flatten() 
        ).all()

        # Rust API is consistent
        assert (
            np.array([
                corrosiff_sr.sum_roi(rois[k-1][p], frames = list(range(p, N_FRAMES ,k)),registration=None)
                for p in range(k)
            ]).T.flatten()
            == corrosiff_sr.sum_roi(
                rois[k-1], frames = list(range(N_FRAMES)), registration=None
            ).flatten()
        ).all()

        # Whole volume agrees
        assert (
            siffc_sr.sum_roi(rois[k-1], frames = list(range(N_FRAMES)), registration=None)
            == corrosiff_sr.sum_roi(rois[k-1], frames = list(range(N_FRAMES)), registration=None)
        ).all()
