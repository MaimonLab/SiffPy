"""
SiffPlotter class for registration dictionaries
"""

from functools import reduce
from typing import Callable, Union

import numpy as np
import holoviews as hv
import operator

from siffpy.siffplot.siffplotter import SiffPlotter
from siffpy.siffplot.utils import *
from siffpy.core.utils.circle_fcns import zeroed_circ

__all__ = [
    'RegistrationPlotter'
]

inherited_params = [
    'local_opts',
    'siffreader'
]

class RegistrationPlotter(SiffPlotter):
    """
    Extends the SiffPlotter functionality to allow
    analysis of binned arrival times of .siff files.
    Discards spatial information.

    Can be initialized with an existing SiffPlotter
    to inherit its properties

    ( e.g. reg_p = RistogramPlotter(siff_plotter)) )
    """
    
    def register(self, *args, **kwargs)->hv.Layout:
        """ Performs registration and then shows output. Takes args and kwargs of self.siffreader.register"""
        self.siffreader.register(*args, **kwargs)
        return self.visualize()

    @apply_opts
    def visualize(self, *args, **kwargs) -> hv.Layout:
        """
        Returns the visualization of the reference frames and
        registration dict of the internal siffreader side-by-side
        """
        if self.reference_frames is None:
            self.reference_frames = self.reference_frames_to_holomap()

        point_plot = self.registration_map()

        ref_zeroed = hv.Dataset(
            (
                range(-int(self.siffreader.im_params.xsize/2), int(self.siffreader.im_params.xsize/2)),
                range(-int(self.siffreader.im_params.ysize/2), int(self.siffreader.im_params.ysize/2)), 
                range(self.siffreader.im_params.num_slices),
                self.siffreader.reference_frames
            ),
            ['x','y','z'], 'Intensity'
        )
        
        ref_holomap = ref_zeroed.to(hv.Image, ['x','y'], 'Intensity', groupby=['z'])
        layout = point_plot + ref_holomap
        layout = layout.opts(
            hv.opts.Image(width = self.siffreader.im_params.xsize, height=self.siffreader.im_params.ysize),
            hv.opts.Points(width = self.siffreader.im_params.xsize, height=self.siffreader.im_params.ysize),
        )
        return layout

    @apply_opts
    def registration_map(self):
        """
        Returns a DynamicMap of the framewise shifts for each z plane
        """
        if self.siffreader.registration_dict is None:
            raise AttributeError("""
                Provided SiffReader object does not have
                a corresponding registration dictionary.

                Please run `register()`, define a `registration_dict`
                attribute, or run `assign_registration_dict(path)` on
                the path to a `.dict` file.
                """
            )
        
        # Now sort out which frames belong to which z_slices

        slice_frame_list = self.siffreader.im_params.framelist_by_slice()

        shifts = np.array([np.append(
            np.array([self.siffreader.registration_dict[frame] for frame in slice_frame_list[idx] ]),
            idx*np.ones((len(slice_frame_list[idx]),1)),
            axis=1
          ) for idx in range(len(slice_frame_list))])


        shifts = np.reshape(shifts,(shifts.shape[0]*shifts.shape[1],3))


        shifts[:,0] = zeroed_circ(shifts[:,0], self.siffreader.im_params.ysize)
        shifts[:,1] = zeroed_circ(shifts[:,1], self.siffreader.im_params.xsize)

        shift_ds = hv.Dataset(shifts, kdims=['y','x','z'])

        point_plot = shift_ds.to(hv.Points,kdims=['x','y'],groupby=['z']).opts(
            xlabel = "X shift (pixels)",
            ylabel = "Y shift (pixels)",
            xlim = (-self.siffreader.im_params.xsize/2, self.siffreader.im_params.xsize/2),
            ylim = (-self.siffreader.im_params.ysize/2, self.siffreader.im_params.ysize/2),
            xticks = [-self.siffreader.im_params.xsize/2, 0, self.siffreader.im_params.xsize/2],
            yticks = [-self.siffreader.im_params.ysize/2, 0, self.siffreader.im_params.ysize/2],
            width = self.siffreader.im_params.xsize,
            height=self.siffreader.im_params.ysize   
        )

        return point_plot
