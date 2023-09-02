"""
Tests the file reader on a very minimal siff file.

TODO: Write me!
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from siffpy import SiffReader
    from siffpy.core.utils import ImParams

import pytest

@pytest.fixture
def test_file_in()->List['SiffReader']:
    """
    Tests that the test file is read in properly.
    """
    from siffpy import SiffReader

    sr = SiffReader()

    assert not sr.opened

    filename = "not_a_real_path.siff"
    try:
        sr = SiffReader(filename)
    except Exception as e:
        assert isinstance(e, FileNotFoundError)

    # In here: test multiple different test files,
    # each with their own compressions / implementations
    # of the various forms.

    filename = "a_real_path.siff"

    sr_raw = SiffReader(filename)
    assert sr_raw.opened

    filename = 'path_to_compressed.siff'
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
