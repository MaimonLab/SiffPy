"""
Tests the file reader on a very minimal siff file.

TODO: Write me!
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from siffpy import SiffReader
    from siffpy.core.utils import ImParams

import pytest

RAW_FILE_NAME = "raw_test"
COMPRESSED_FILE_NAME = "compressed_test"

@pytest.fixture(scope='session')
def load_test_files(tmp_path_factory, request)->List[str]:
    """
    Loads test files from the specified url into a temp
    directory for use in other tests
    """

    import shutil
    import tifffile
    from pathlib import Path

    # Create a temporary directory, install
    # a file from the server to it.
    tmp_dir = tmp_path_factory.mktemp("test_siff_files")

    filename = request.module.__file__
    test_dir = Path(filename).with_suffix("")

    # TODO copy the data over correctly, and from an online source!!!
    shutil.copy(
        (test_dir / RAW_FILE_NAME).with_suffix('.siff'),
        (tmp_dir / RAW_FILE_NAME).with_suffix('.siff')
    )

    shutil.copy(
        (test_dir / COMPRESSED_FILE_NAME).with_suffix('.siff'),
        (tmp_dir / COMPRESSED_FILE_NAME).with_suffix('.siff')
    )


    # shutil.copy(
    #     test_dir / "seed_mus.npy",
    #     tmp_dir / "seed_mus.npy"
    # )

    return tmp_dir


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

    filename = (load_test_files / RAW_FILE_NAME).with_suffix('.siff')

    sr_raw = SiffReader(filename)
    assert sr_raw.opened

    filename = (load_test_files / COMPRESSED_FILE_NAME).with_suffix('.siff')
    sr_compressed = SiffReader(filename)

    assert sr_compressed.opened

    return [sr_raw, sr_compressed]

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

    sr_raw, sr_compressed = test_file_in

    check_imparams(sr_raw.im_params, 1, 1, 1, 1)
    check_imparams(sr_compressed.im_params, 1, 1, 1, 1)

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