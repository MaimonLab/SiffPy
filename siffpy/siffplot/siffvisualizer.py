"""

Class and related functions for the SiffVisualizer, a class
which produces fast display of raw fluorescence or FLIM images,
when coupled to a SiffReader object. These data are not analyzed,
though it does permit some adjustment of visualization parameters.

SCT 09/23/2021
"""
import holoviews as hv

from ..siffpy import SiffReader

class SiffVisualizer():
    """
    A class that permits visualization of fluorescence
    or FLIM images using HoloViews DynamicMap objects.

    This allows dynamically reading data from disk and
    displaying it as collections of images. My intention
    is to make this interface with SiffPlotter objects at
    some point (as long as they share a ref to the same SiffReader)
    and allow visualization of features highlighted by the
    SiffPlotter.
    
    """
    def __init__(self, siffreader : SiffReader):
        self.siffreader = siffreader