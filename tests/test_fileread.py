"""
Tests the file reader on a very minimal siff file.

TODO: Write me!
"""

from typing import TYPE_CHECKING, List, Callable, Any, Tuple
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from siffpy import SiffReader
    from siffpy.core.utils import ImParams

import pytest

def download_files_from_dropbox(local_path : Path):
    """
    Accesses the .siff files from the shared link
    on Dropbox. Short-to-medium term filesharing
    solution
    """
    import os
    from dropbox import Dropbox
    import dropbox

    DROPBOX_SECRET_TOKEN = os.environ['DROPBOX_SECRET']
    DROPBOX_APP_KEY = os.environ['DROPBOX_APP_KEY']
    REFRESH_TOKEN = os.environ['DROPBOX_REFRESH_TOKEN']
    SHARED_LINK = os.environ['DROPBOX_SHARED_LINK']

    dbx = Dropbox(app_key= DROPBOX_APP_KEY, app_secret=DROPBOX_SECRET_TOKEN, oauth2_refresh_token=REFRESH_TOKEN)

    dbx.check_and_refresh_access_token()
    link = dropbox.files.SharedLink(url=SHARED_LINK)

    for x in dbx.files_list_folder('', shared_link=link).entries:
        meta, response = dbx.sharing_get_shared_link_file(link.url, path = f'/{x.name}')
        with open(local_path / meta.name, 'wb') as f:
            f.write(response.content)

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
    # Create a temporary directory, install
    # a file from the server to it.
    tmp_dir = tmp_path_factory.mktemp("test_siff_files")
    download_files_from_dropbox(tmp_dir)

    return tmp_dir


@pytest.fixture(scope = 'function')
def test_file_in(load_test_files : Path)->List['SiffReader']:
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
        assert isinstance(e, (FileNotFoundError, OSError))

    # In here: test multiple different test files,
    # each with their own compressions / implementations
    # of the various forms.
    #filename = (load_test_files / RAW_FILE_NAME).with_suffix('.siff')

    # sr_raw = SiffReader(filename)
    # assert sr_raw.opened

    # return [sr_raw]

    readers_and_meta = []
    for file in load_test_files.glob('*'):
        if file.suffix == '.siff':
            sr = SiffReader(file)
            assert sr.opened
            readers_and_meta.append(sr)

    return readers_and_meta
        

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
        framelist = sr.im_params.flatten_by_timepoints()
        sr.get_frames(framelist)
        sr.get_frames(framelist, registration_dict = {})
        try:
            sr.get_frames(framelist, registration_dict = {0 : (0, 0)})
        except Exception as e:
            assert isinstance(e, ValueError)
        else:
            assert False
        sr.get_frames(framelist, registration_dict = {k : (0, 0) for k in framelist})

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
        
        assert (np.array([
            sr.sum_mask(mask, registration_dict = {}) for mask in two_d_masks
            ]) == sr.sum_masks(two_d_masks, registration_dict = {})).all()
        
        # Do the same for 3d
        assert (np.array([
            sr.sum_mask(mask, registration_dict = {}) for mask in three_d_masks
            ]) == sr.sum_masks(three_d_masks, registration_dict= {})).all()

    apply_test_to_all(test_reader, test_file_in, masks)

def test_flim_methods(test_file_in : List['SiffReader']):
    """
    Tests that the flim methods work properly
    """
    pass

def test_siff_to_tiff():
    pass