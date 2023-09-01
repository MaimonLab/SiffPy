from typing import Any, Tuple, List, Dict, Optional, TYPE_CHECKING

import numpy as np

BOOL_ARRAY = np.ndarray[Any, np.dtype[np.bool_]]
UINT16_ARRAY = np.ndarray[Any, np.dtype[np.uint16]]
UINT64_ARRAY = np.ndarray[Any, np.dtype[np.uint64]]
FLOAT_ARRAY = np.ndarray[Any, np.dtype[np.float64]]

if TYPE_CHECKING:
    from siffpy.core import FLIMParams

class FrameData():

    @property
    def imageWidth(self)->int:...

    @property
    def imageHeight(self)->int:...

class SiffIO():
    """
    This class is a wrapper for the C++ SiffReader class.

    Controls file reading and formats data streams into
    numpy arrays.
    """
    @property
    def filename(self)->str:...

    @property
    def debug(self, debug_status : bool)->bool:...

    @property
    def status(self)->str:...

    def open(self, filename : str)->None:...

    def close(self)->None:...

    def get_file_header(self)->Dict:...

    def num_frames(self)->int:...

    def get_frames(
        self,
        frames : List[int],
        registration : Dict = {},
        as_array : bool = True,
    )->UINT16_ARRAY:...

    def get_frame_metadata(self, frames : List[int] = [])->List[Dict]:...

    def pool_frames(
        self,
        frames : List[int],
        flim : bool = False,
        registration : Optional[Dict] = None,
    )->UINT16_ARRAY:
        """ NOT IMPLEMENTED """

    def flim_map(
        self,
        params : 'FLIMParams',
        frames : List[int],
        confidence_metric : str = 'chi_sq',
        registration : Optional[Dict] = None,
    )->Tuple[FLOAT_ARRAY, UINT16_ARRAY, FLOAT_ARRAY]:
        """
        Returns a tuple of (flim_map, intensity_map, confidence_map)
        where flim_map is the empirical lifetime with the offset of
        params subtracted.
        """
        ...

    def sum_roi(
        self,
        mask : BOOL_ARRAY,
        frames : Optional[List[int]] = None,
        registration : Optional[Dict] = None,
    )->UINT16_ARRAY:
        """
        Mask may have more than 2 dimensions, but
        if so then be aware that the frames will be
        iterated through sequentially, rather than
        aware of the correspondence between frame
        number and mask dimension. Returns a 1D
        arrary of the same length as the frames
        provided, regardless of mask shape.
        """

    def sum_roi_flim(
        self,
        mask : BOOL_ARRAY,
        params : 'FLIMParams',
        frames : Optional[List[int]] = None,
        registration : Optional[Dict] = None,
    )->UINT16_ARRAY:
        """
        Mask may have more than 2 dimensions, but
        if so then be aware that the frames will be
        iterated through sequentially, rather than
        aware of the correspondence between frame
        number and mask dimension. Returns a 1D
        arrary of the same length as the frames
        provided, regardless of mask shape.
        """

    def get_histogram(self,frames : Optional[List[int]] = None,)->UINT64_ARRAY:
        """
        Returns a histogram of the arrival times of photons in the
        frames provided. If no frames are provided, then the
        histogram is of all the frames in the file.    
        """

    def get_experiment_timestamps(self, frames : Optional[List[int]] = None,)->FLOAT_ARRAY:
        """
        Returns an array of timestamps of each frame based on
        counting laser pulses since the start of the experiment.

        Units are SECONDS.

        Extremely low jitter, small amounts of drift (maybe 50 milliseconds an hour).
        """

    def get_epoch_timestamps_laser(self, frames : Optional[List[int]] = None,)->UINT64_ARRAY:
        """
        Returns an array of timestamps of each frame based on
        counting laser pulses since the start of the experiment.
        Extremely low jitter, small amounts of drift (maybe 50 milliseconds an hour).

        Can be corrected using get_epoch_timestamps_system.
        """

    def get_epoch_timestamps_system(self, frames : Optional[List[int]] = None,)->UINT64_ARRAY:
        """
        Returns an array of timestamps of each frame based on
        the system clock. High jitter, but no drift.

        WARNING if system timestamps do not exist, the function
        will CRASH.
        """

    def get_epoch_both(self, frames : Optional[List[int]] = None,)->UINT64_ARRAY:
        """
        Returns an array containing both the laser timestamps
        and the system timestamps.

        The first row is laser timestamps, the second
        row is system timestamps.

        WARNING if system timestamps do not exist, the function
        will CRASH.
        """
    
def suppress_warnings()->None:...

def report_warnings()->None:...

def debug()->None:...

def siff_to_tiff(sourcepath : str, savepath : Optional[str] = None)->None:...