"""

Class and related functions for the SiffVisualizer, a class
which produces fast display of raw fluorescence or FLIM images,
when coupled to a SiffReader object. These data are not typically
quantitatively analyzed, though it does permit some adjustment of
visualization parameters.

SCT 09/23/2021
"""
import logging, pickle, os
from pathlib import Path

from siffpy.core import SiffReader
from siffpy.siffplot.roi_protocols import rois
from siffpy.siffplot.utils.exceptions import *
from siffpy.siffplot.napari_viewers import FrameViewer

class SiffVisualizer():
    """
    A class that permits visualization of fluorescence
    or FLIM images using napari.

    This allows dynamically reading data from disk and
    displaying it as collections of images. My intention
    is to make this interface with SiffPlotter objects at
    some point (as long as they share a ref to the same SiffReader)
    and allow visualization of features highlighted by the
    SiffPlotter.
    
    The SiffVisualizer object is an interface for the
    underlying plotting libraries, not a plotter itself.

    """
    def __init__(self, siffreader : SiffReader):
        self.siffreader = siffreader
        self.visual = None
        self.image_opts = {
            'yaxis' : None,
            'xaxis' : None,
            'cmap' : 'greys_r',
            'clim' : (None,None),
            'invert_yaxis' : False,
        }
        self.loaded_frames = False
        self._rois = []

        directory_with_file_name = Path(self.siffreader.filename).with_suffix("")

        if directory_with_file_name.exists():
            if any([file.endswith('.roi') for file in os.listdir(directory_with_file_name)]):
                logging.warning("Found .roi file(s) in directory with open file.\nLoading ROI(s)")
                self.load_rois(path = directory_with_file_name)

    def view_frames(self, load_frames : bool = False, **kwargs):
        """
        Returns a napari Viewer object loaded with frames to image. Accepts the FrameViewer `NapariInterface` subclass's
        initialization arguments as keyword arguments (e.g. batch_fcn).

        Arguments
        ---------

        load_frames : bool (optional)

            Pre-load all of the frames (takes longer to return and occupies a large chunk of RAM,
            but then it's free from the SiffReader object and can be used while that's busy).

        NOTE: If load_frames is used, you'll need to pre-determine the pool_width by adding the
        kwarg pool_width (type int). Default is 1.

        Returns
        -------
        v : siffpy.siffplot.napari_viewers.FrameViewer

            A FrameViewer that subclasses the NapariInterface
            class to match typical functionality for viewing
            individual frames of image data.
        """
        
        self.frames = FrameViewer(self.siffreader, load_frames = load_frames, image_opts = self.image_opts, **kwargs)
        
        return self.frames
    
    @property
    def rois(self):
        """ I regret the decision of making this either a list or a None or an ROI"""
        if len(self._rois) == 0:
            return None
        if len(self._rois) == 1:
            return self._rois[0]
        return self._rois
    
    @rois.setter
    def rois(self, value):
        self._rois = value

    def add_roi(self, roi):
        """
        Adds an ROI to the list of ROIs to be displayed.
        """
        if self.rois is None:
            self.rois = [roi]
        else:
            self.rois = list(self.rois) + [roi]

    ## ROIS
    def save_rois(self, path : str = None):
        """
        Saves the rois stored in the self.rois attribute. The default path is in the directory with
        the .siff file.

        Arguments
        ---------
        
        path : str (optional)

            Where to save the ROIs.
        """
        if len(self.rois) == 0:
            raise NoROIException("SiffVisualizer object has no ROIs stored")
        
        if path is None:
            if not self.siffreader.opened:
                raise RuntimeError("Siffreader has no open file, and no alternative path was provided.")
            path = Path(self.siffreader.filename).with_suffix("")

        path = Path(path).with_suffix("")

        if hasattr(self.rois,'__iter__'):
            for roi in self.rois:
                roi.save(path)
        elif not (self.rois is None):
            # else, just save the one.
            self.rois.save(path)
        else:
            raise NoROIException("No attribute rois defined for this SiffVisualizer.")

    def load_rois(self, path : str = None):
        """
        Loads rois stored at location 'path'. If no path is specified, then
        it looks for .roi files matching the file name
        """

        if path is None:
            path = Path(self.siffreader.filename).with_suffix("")
        path = Path(path)

        roi_files = [path/file for file in os.listdir(str(path)) if file.endswith('.roi')]

        self.rois = []
        for roi in roi_files:
            with open(roi, 'rb') as curr_file:
                if type(self.rois) is list:
                    self.rois.append(pickle.load(curr_file))
                elif issubclass(type(self.rois),rois.ROI):
                    self.rois = list(self.rois) + [pickle.load(curr_file)]
                else:
                    self.rois = [pickle.load(curr_file)]