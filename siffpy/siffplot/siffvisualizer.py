"""

Class and related functions for the SiffVisualizer, a class
which produces fast display of raw fluorescence or FLIM images,
when coupled to a SiffReader object. These data are not analyzed,
though it does permit some adjustment of visualization parameters.

SCT 09/23/2021
"""
from typing import Iterable
import holoviews as hv
hv.extension('bokeh')
import numpy as np

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
        self.visual = None

    def view_frames(self, z_planes : list[int] = None, color : int = 0, **kwargs) -> hv.DynamicMap:
        """
        Returns a dynamic map object that permits visualization
        of individual timepoints across z-planes, or restricting
        z-plane.

        Arguments
        ---------

        z_planes : list[int] (optional)

            Which z-planes to show, 0-indexed. Defaults to all.

        color : int or list[int] (optional)

            Which color to show. 0 (first channel) by default.

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

        IM_OPTS = {
            'yaxis' : None,
            'xaxis' : None,
            'cmap' : 'greys_r',
            'clim' : (0,1)
        }

        def show_frames(t_val, pool_width):
            frames = self.siffreader.sum_across_time(
                timepoint_start = t_val,
                timepoint_end = (t_val+1) + pool_width,
                timespan = pool_width,
                z_list = z_planes,
                color_list = color,
                registration_dict = self.siffreader.registrationDict
            )
            
            images = [hv.Image(frames[j]).opts(**IM_OPTS) for j in range(1,len(frames))]
            
            return sum(images, hv.Image(frames[0]).opts(**IM_OPTS)).cols(int(np.sqrt(len(z_planes)))+1)

        hv.output(widget_location='top') # may start doing this with panel at some point in the future?

        dm : hv.DynamicMap = hv.DynamicMap(show_frames, kdims = ['timepoint', 'pool_width'])
        dm = dm.redim.range(
            timepoint=(0,self.siffreader.im_params.num_frames//self.siffreader.im_params.frames_per_volume),
            pool_width=(1,20)
        )
        dm = dm.redim.type(timepoint=int, pool_width = int).redim.step(timepoint=1, pool_width = 1)

        return dm
