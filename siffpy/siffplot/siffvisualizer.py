"""

Class and related functions for the SiffVisualizer, a class
which produces fast display of raw fluorescence or FLIM images,
when coupled to a SiffReader object. These data are not analyzed,
though it does permit some adjustment of visualization parameters.

SCT 09/23/2021
"""
import math

from typing import Iterable
import holoviews as hv
import numpy as np
import functools, operator, logging

from ..siffpy import SiffReader

NAPARI = False
try:
    import napari
    import dask
    from .napari_viewers import FrameViewer
    NAPARI = True
except ImportError:
    hv.extension('bokeh') # no need to do this unless
    # we're defaulting to holoviews, just because there
    # is some headache with napari and hv at the moment.


class SiffVisualizer():
    """
    A class that permits visualization of fluorescence
    or FLIM images using HoloViews DynamicMap objects
    or napari.

    This allows dynamically reading data from disk and
    displaying it as collections of images. My intention
    is to make this interface with SiffPlotter objects at
    some point (as long as they share a ref to the same SiffReader)
    and allow visualization of features highlighted by the
    SiffPlotter.
    
    The SiffVisualizer object is an interface for the
    underlying plotting libraries, not a plotter itself.

    """
    def __init__(self, siffreader : SiffReader, backend : str = 'napari'):
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
        if not backend in ['napari', 'holoviews']:
            if NAPARI:
                logging.warn(f"Invalid backend argument {backend}. Defaulting to napari.")
                backend = 'napari'
            else:
                logging.warn(f"Invalid backend argument {backend}. Defaulting to holoviews.")
                backend = 'holoviews'
        self.backend = backend

    def view_frames(self, z_planes : list[int] = None, color : int = 0, load_frames : bool = False, **kwargs):
        """
        Function used to visualize data provided. Calls appropriate backend viewing functions. If backend is
        HoloViews, returns a DynamicMap object that allows scrolling through timepoints across z planes.
        
        If backend is napari, returns a napari viewer object.
        """
        if self.backend == 'holoviews':
            return self.view_frames_hv(z_planes = z_planes, color = color, load_frames = load_frames, **kwargs)
        if self.backend == 'napari':
            return self.view_frames_napari(z_planes = z_planes, color = color, load_frames = load_frames, **kwargs)
        raise AttributeError("Specified backend is not valid.")


    def view_frames_hv(self, z_planes : list[int] = None, color : int = 0, load_frames : bool = False, **kwargs) -> hv.DynamicMap:
        """
        Returns a dynamic map object that permits visualization
        of individual timepoints across z-planes, or restricting
        z-plane.

        Adjusting the SiffVisualizer's image_opts attribute's keys will change how this is plotted.

        Arguments
        ---------

        z_planes : list[int] (optional)

            Which z-planes to show, 0-indexed. Defaults to all.

        color : int or list[int] (optional)

            Which color to show. 0 (first channel) by default.

        load_frames : bool (optional)

            Pre-load all of the frames (takes longer to return and occupies a large chunk of RAM,
            but then it's free from the SiffReader object and can be used while that's busy).

        NOTE: If load_frames is used, you'll need to pre-determine the pool_width by adding the
        kwarg pool_width (type int). Default is 1.

        Returns
        -------
        dm : hv.DynamicMap

            A DynamicMap object which reads the linked .siff file and
            displays frames from it.
        """
        if not self.siffreader.opened:
            raise RuntimeError("SiffReader object not yet initialized by opening a file")

        if z_planes is None:
            z_planes = list(range(self.siffreader.im_params.num_slices))

        if not isinstance(color, Iterable):
            color = [color]
        else:
            color = list(color)
        
        loaded_frames = False
        if load_frames:
            pool_width = 1
            if 'pool_width' in kwargs:
                pool_width = kwargs['pool_width']

            self.frames = frames = self.siffreader.sum_across_time(
                    timepoint_start = 0,
                    timepoint_end = self.siffreader.im_params.num_frames // self.siffreader.im_params.frames_per_volume , # number of volumes
                    timespan = pool_width,
                    z_list = z_planes,
                    color_list = color,
                    registration_dict = self.siffreader.registrationDict
                )
            loaded_frames = True

        def show_frames(t_val, pool_width):
            # local function def
            if not self.loaded_frames:
                frames = self.siffreader.sum_across_time(
                    timepoint_start = t_val,
                    timepoint_end = t_val + pool_width,
                    timespan = pool_width,
                    z_list = z_planes,
                    color_list = color,
                    registration_dict = self.siffreader.registrationDict
                )
            else:
                frames = [self.frames[t_val*self.siffreader.frames_per_volume + k] for k in range(len(z_planes))]

            images = [hv.Image(frames[j]).opts(**(self.image_opts)) for j in range(0,len(frames))]
            
            return functools.reduce(operator.add, images).cols(math.floor(np.sqrt(len(z_planes)))) # make a Layout by adding each frame

        hv.output(widget_location='top') # may start doing this with panel at some point in the future?
        
        if loaded_frames:
            dm : hv.DynamicMap = hv.DynamicMap(lambda t: show_frames(t,1), kdims = ['timepoint'])
            dm = dm.redim.range(
                timepoint=(0,self.siffreader.im_params.num_frames//self.siffreader.im_params.frames_per_volume)
            )
            dm = dm.redim.type(timepoint=int).redim.step(timepoint=1)

        else:
            dm : hv.DynamicMap = hv.DynamicMap(show_frames, kdims = ['timepoint', 'pool_width'])
            dm = dm.redim.range(
                timepoint=(0,self.siffreader.im_params.num_frames//self.siffreader.im_params.frames_per_volume),
                pool_width=(1,20)
            )
            dm = dm.redim.type(timepoint=int, pool_width = int).redim.step(timepoint=1, pool_width = 1)
        self.visual = dm
        return self.visual

    def view_frames_napari(self, z_planes : list[int] = None, color : int = 0, load_frames : bool = False, **kwargs):
        """
        Returns a napari Viewer object loaded with frames to image

        Arguments
        ---------

        z_planes : list[int] (optional)

            Which z-planes to show, 0-indexed. Defaults to all.

        color : int or list[int] (optional)

            Which color to show. 0 (first channel) by default.

        load_frames : bool (optional)

            Pre-load all of the frames (takes longer to return and occupies a large chunk of RAM,
            but then it's free from the SiffReader object and can be used while that's busy).

        NOTE: If load_frames is used, you'll need to pre-determine the pool_width by adding the
        kwarg pool_width (type int). Default is 1.

        Returns
        -------
        v : siffpy.siffplot.napari_viewers.FrameViewer

            A FrameViewer that subclasses the napari.Viewer
            class to match typical functionality for viewing
            individual frames of image data.
        """
        
        self.visual = FrameViewer(self.siffreader, load_frames = load_frames, image_opts = self.image_opts)
        
        return self.visual
