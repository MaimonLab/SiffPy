from typing import Any, Union

import numpy as np

from siffpy.core import FLIMParams

class FrameData():

    @property
    def imageWidth(self)->int:...

    @property
    def imageHeight(self)->int:...

class SiffIO():

    @property
    def filename(self)->str:...

    @property
    def debug(self, debug_status : bool)->bool:...

    @property
    def status(self)->str:...

    def open(self, filename : str)->None:...

    def close(self)->None:...

    def get_file_header(self)->dict:...

    def num_frames(self)->int:...

    def get_frames(
        self,
        frames : list[int],
        registration : dict = {},
        as_array : bool = True,
    )->np.ndarray:...

    def get_frame_metadata(self, frames : list[int] = [])->list[dict]:...

    def pool_frames(
        self,
        frames : list[int],
        flim : bool = False,
        registration : dict = None,
    )->np.ndarray:
        """ NOT IMPLEMENTED """

    def flim_map(
        self,
        params : FLIMParams,
        frames : list[int],
        confidence_metric : str = 'chi_sq',
        registration : dict = None,
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a tuple of (flim_map, intensity_map, confidence_map)
        where flim_map is the empirical lifetime with the offset of
        params subtracted.
        """
        ...

    def sum_roi(
        self,
        mask : np.ndarray,
        frames : list[int] = None,
        registration : dict = None,
    )->np.ndarray:
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
        mask : np.ndarray,
        params : FLIMParams,
        frames : list[int] = None,
        registration : dict = None,
    )->np.ndarray:
        """
        Mask may have more than 2 dimensions, but
        if so then be aware that the frames will be
        iterated through sequentially, rather than
        aware of the correspondence between frame
        number and mask dimension. Returns a 1D
        arrary of the same length as the frames
        provided, regardless of mask shape.
        """

    def get_histogram(self,frames : list[int] = None,)->np.ndarray:...

def suppress_warnings()->None:...

def report_warnings()->None:...

def debug()->None:...

def siff_to_tiff(sourcepath : str, savepath : str = None)->None:...