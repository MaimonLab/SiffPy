"""
Tests the file reader on some very minimal siff files.

TODO: Write me!
"""

from typing import TYPE_CHECKING, List, Callable, Any, Tuple
import numpy as np

if TYPE_CHECKING:
    from siffpy import SiffReader
    from siffpy.core.utils import ImParams

import pytest

RAW_FILE_NAME = "raw_test"
COMPRESSED_FILE_NAME = "compressed_test"

# Short term for debugging.

# Problematic file with one extra frame that wasn't finished
# saving TODO make sure this gets tested too!!
#TEST_FILE_DIR = '/Users/stephen/Desktop/Data/imaging'
#TEST_FILE_NAME = '2024_06_03_9.tiff'

# Functional file
TEST_FILE_DIR = '/Users/stephen/Desktop/Data/imaging/2024-05/2024-05-27/R41H07_greenCamui_alpha/Fly1/'
TEST_FILE_NAME = 'FB_1.siff'

def apply_test_to_all(
        test : Callable[[Tuple[List['SiffReader'],...]], Any],
        *filewise_args : Tuple[List['SiffReader'],...],
    )->None:
    """
    Applies a test to all the test files (and other args
    if passed along)
    """
    for args in zip(*filewise_args):
        test(*args)


@pytest.fixture(scope='session')
def load_test_files(tmp_path_factory, request)->List[str]:
    """
    Loads test files from the specified url into a temp
    directory for use in other tests
    """

    #import shutil
    from pathlib import Path

    # Create a temporary directory, install
    # a file from the server to it.
    #tmp_dir = tmp_path_factory.mktemp("test_siff_files")

    #filename = request.module.__file__
    #test_dir = Path(filename).with_suffix("")
    test_dir = Path(TEST_FILE_DIR)

    # shutil.copy(
    #     test_dir / TEST_FILE_NAME,
    #     tmp_dir / TEST_FILE_NAME
    # )

    # TODO copy the data over correctly, and from an online source!!!
    # shutil.copy(
    #     (test_dir / RAW_FILE_NAME).with_suffix('.siff'),
    #     (tmp_dir / RAW_FILE_NAME).with_suffix('.siff')
    # )

    # shutil.copy(
    #     (test_dir / COMPRESSED_FILE_NAME).with_suffix('.siff'),
    #     (tmp_dir / COMPRESSED_FILE_NAME).with_suffix('.siff')
    # )

    #return tmp_dir
    return test_dir


@pytest.fixture(scope = 'function')
def test_file_in(load_test_files)->List['SiffReader']:
    """
    Tests that the test file is read in properly.
    """
    from siffpy import SiffReader

    sr = SiffReader()

    assert not sr.opened

    filename = load_test_files / "not_a_real_file.siff"
    try:
        sr = SiffReader(filename)
    except Exception as e:
        assert isinstance(e, FileNotFoundError)

    # In here: test multiple different test files,
    # each with their own compressions / implementations
    # of the various forms.

    filename = load_test_files / TEST_FILE_NAME

    #filename = (load_test_files / RAW_FILE_NAME).with_suffix('.siff')

    sr_raw = SiffReader(filename)
    assert sr_raw.opened

    return [sr_raw]

    # filename = (load_test_files / COMPRESSED_FILE_NAME).with_suffix('.siff')
    # sr_compressed = SiffReader(filename)

    # assert sr_compressed.opened

    # return [sr_raw, sr_compressed]

def test_imparams(test_file_in : List['SiffReader']):
    """
    Tests that the image parameters are read in properly.
    """

    def check_imparams(
        im_params : 'ImParams',
        num_slices : int,
        num_colors : int,
        num_frames : int,
        num_true_frames : int,
        ):
        """
        Tests that the various scanimage parameters
        are read in properly.

        TODO: add flyback checks etc
        """
        assert im_params.num_slices == num_slices
        assert im_params.num_colors == num_colors
        assert im_params.num_frames == num_frames
        assert im_params.num_true_frames == num_true_frames

    sr = test_file_in[0]

    assert sr.im_params is not None

    #sr_raw, sr_compressed = test_file_in

    # check_imparams(sr_raw.im_params, 1, 1, 1, 1)
    # check_imparams(sr_compressed.im_params, 1, 1, 1, 1)

def test_metadata(test_file_in : List['SiffReader']):
    """
    Tests that the metadata C++ calls work properly
    """
    pass

    # def check_metadata(
    #     metadata : dict,
    #     num_slices : int,
    #     num_colors : int,
    #     num_frames : int,
    #     num_true_frames : int,
    #     ):
    #     """
    #     Tests that the various scanimage parameters
    #     are read in properly.

def test_read_time(test_file_in : List['SiffReader']):
    """
    Tests that the time methods run at all -- first
    at the `SiffIO` level then at the `SiffReader` level.
    """

    def test_reader(sr : 'SiffReader'):
        """
        For now, this just checks that these
        methods run without killing the kernel.

        TODO: Add some tests to make sure the numbers
        are right!
        """
        sr.siffio.get_epoch_timestamps_laser(frames=list(range(100)))
        sr.siffio.get_epoch_timestamps_system(frames=list(range(100)))
        sr.siffio.get_experiment_timestamps(frames=list(range(100)))
        sr.siffio.get_epoch_both(frames=list(range(100)))

        sr.siffio.get_epoch_timestamps_laser()
        sr.siffio.get_epoch_timestamps_system()
        sr.siffio.get_experiment_timestamps()
        sr.siffio.get_epoch_both()

        sr.t_axis()
        sr.t_axis(reference_time = 'epoch')

    apply_test_to_all(test_reader, test_file_in,)

def test_get_frames(test_file_in : List['SiffReader']):
    """
    Tests that the frame methods work properly
    """
    def test_reader(sr: 'SiffReader'):
        """ Tests frame reading methods """
        sr.get_frames(sr.im_params.flatten_by_timepoints())

    apply_test_to_all(test_reader, test_file_in,)

@pytest.fixture(scope = 'function')
def masks(test_file_in : List['SiffReader'])->List[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Returns a list of masks for testing -- one of 2d arrays and one of 3d arrays.
    """

    def produce_masklist_2d(sr : 'SiffReader'):
        """ Produces a mask for testing """
        base_mask = np.zeros(sr.im_params.shape).astype(bool)

        # Only true in one quadrant
        base_mask[:base_mask.shape[0]//2, :base_mask.shape[1]//2] = True

        masks = [np.roll(base_mask, idx*sr.im_params.xsize//10, axis = 0) for idx in range(10)]
        return masks

    def produce_masklist_3d(sr : 'SiffReader'):

        base_mask = np.zeros(sr.im_params.single_channel_volume).astype(bool)

        # Set one stripe to true

        base_mask[..., :base_mask.shape[2]//8] = True

        # True in rolling stripes in the x dimension
        masks = [
            np.roll(base_mask, idx*sr.im_params.xsize//10, axis = 0) for idx in range(10)
        ]

        return masks
    
    return [
        (
            produce_masklist_2d(sr),
            produce_masklist_3d(sr)
        ) for sr in test_file_in
    ]

def test_mask_intensity_methods(
        test_file_in : List['SiffReader'],
        masks : List[Tuple[List[np.ndarray]]]
    ):
    """
    Tests that the mask methods work properly
    """

    def test_reader(
        sr : 'SiffReader',
        corresponding_masks : Tuple[List[np.ndarray]],
        ):
        """ Tests mask methods """

        two_d_masks, three_d_masks = corresponding_masks

        # Confirm that the list of 2d masks gives the same
        # as making an array from the returned value of
        # each passed one at a time.

        assert (np.array([
            sr.sum_mask(mask) for mask in two_d_masks
            ]) == sr.sum_masks(two_d_masks)).all()
        
        # Do the same for 3d
        assert (np.array([
            sr.sum_mask(mask) for mask in three_d_masks
            ]) == sr.sum_masks(three_d_masks)).all()

    apply_test_to_all(test_reader, test_file_in, masks)

def test_flim_methods(
        test_file_in : List['SiffReader'],
        masks : List[Tuple[List[np.ndarray]]]
    ):
    """
    Tests that the flim methods work properly.
    Doesn't need correct FLIMParams, just to have
    them at all!
    """
    from siffpy.core.flim import FLIMParams, Exp, Irf
    from siffpy.siffmath import FlimTrace

    test_params = FLIMParams(
        Exp(tau = 1.1, frac = 0.3, units = 'nanoseconds'),
        Exp(tau = 2.5, frac = 0.7, units='nanoseconds'),
        Irf(mean = 1.3, sigma = 0.02, units = 'nanoseconds'),
    )

    def test_reader(
        sr : 'SiffReader',
    ):
        """ Test the unmasked FLIM methods """
        sr.get_frames_flim(
            test_params,
            frames = sr.im_params.flatten_by_timepoints()
        )

    def test_reader_and_masks(
        sr : 'SiffReader',
        corresponding_masks : Tuple[List[np.ndarray]],
        ):
        """ Tests flim mask methods agree """

        two_d_masks, three_d_masks = corresponding_masks

        # Confirm that the list of 2d masks gives the same
        # as making an array from the returned value of
        # each passed one atlim a time.

        assert (FlimTrace([
            sr.sum_mask_flim(test_params, mask) for mask in two_d_masks
            ]) == sr.sum_masks_flim(test_params, two_d_masks)).all()
        
        # Do the same for 3d
        assert (FlimTrace([
            sr.sum_mask_flim(test_params, mask) for mask in three_d_masks
            ]) == sr.sum_masks_flim(test_params, three_d_masks)).all()
        
    apply_test_to_all(test_reader, test_file_in,)
    apply_test_to_all(test_reader_and_masks, test_file_in, masks)

def test_siff_to_tiff():
    pass