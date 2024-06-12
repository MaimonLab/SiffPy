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


def test_flim_params(load_test_files: List[str])->None:
    """
    Tests the FLIMParams class methods
    """
    pass