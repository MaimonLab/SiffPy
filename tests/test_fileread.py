"""
Tests the file reader on a very minimal siff file.

TODO: Write me!
"""

from typing import TYPE_CHECKING, List, Callable, Any

if TYPE_CHECKING:
    from siffpy import SiffReader
    from siffpy.core.utils import ImParams

import pytest

RAW_FILE_NAME = "raw_test"
COMPRESSED_FILE_NAME = "compressed_test"

# Short term for debugging.

# Problematic file with one extra frame
TEST_FILE_DIR = '/Users/stephen/Desktop/Data/imaging'
TEST_FILE_NAME = '2024_06_03_9.tiff'

def apply_test_to_all(
        test_files : List['SiffReader'],
        test : Callable[['SiffReader'], Any]
    ):
    """
    Applies a test to all the test files.
    """
    all(test(sr) for sr in test_files)


@pytest.fixture(scope='session')
def load_test_files(tmp_path_factory, request)->List[str]:
    """
    Loads test files from the specified url into a temp
    directory for use in other tests
    """

    import shutil
    from pathlib import Path

    # Create a temporary directory, install
    # a file from the server to it.
    tmp_dir = tmp_path_factory.mktemp("test_siff_files")

    #filename = request.module.__file__
    #test_dir = Path(filename).with_suffix("")
    test_dir = Path(TEST_FILE_DIR)
    TEST_FILE_NAME = '2024_06_03_9.tiff'

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

    apply_test_to_all(test_file_in, test_reader)

def test_get_frames(test_file_in : List['SiffReader']):
    """
    Tests that the frame methods work properly
    """
    def test_reader(sr: 'SiffReader'):
        sr.get_frames(sr.im_params.flatten_by_timepoints())

    apply_test_to_all(test_file_in, test_reader)

def test_mask_methods(test_file_in : List['SiffReader']):
    """
    Tests that the mask methods work properly
    """
    pass

def test_flim_methods(test_file_in : List['SiffReader']):
    """
    Tests that the flim methods work properly
    """
    pass

def test_siff_to_tiff():
    pass