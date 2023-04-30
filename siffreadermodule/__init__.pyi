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

    def get_metadata(self, frames : list[int] = [])->dict:...

    def pool_frames(
        self,
        frames : list[int],
        flim : bool = False,
        registration : dict = None,
    )->np.ndarray:...

    def flim_map(
        self,
        params : FLIMParams,
        frames : list[int],
        confidence_metric : str = 'chi_sq',
        registration : dict = None,
    )->np.ndarray:...

    def sum_roi(
        self,
        mask : np.ndarray,
        frames : list[int] = None,
        registration : dict = None,
    )->np.ndarray:...

    def sum_roi_flim(
        self,
        mask : np.ndarray,
        params : FLIMParams,
        frames : list[int] = None,
        registration : dict = None,
    )->np.ndarray:...

    def get_histogram(self,frames : list[int] = None,)->np.ndarray:...

def suppress_warnings()->None:...

def report_warnings()->None:...

def debug()->None:...

def siff_to_tiff(sourcepath : str, savepath : str = None)->None:...