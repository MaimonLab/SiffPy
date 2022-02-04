"""
SiffPlotter class for arrival time histograms
"""

from functools import reduce
from typing import Callable, Union

import numpy as np
import holoviews as hv
import operator

from ...siffplot.siffplotter import SiffPlotter, apply_opts
from ...siffutils import FLIMParams
from ..utils import *
from ...siffutils.slicefcns import *

__all__ = [
    'HistogramPlotter'
]

inherited_params = [
    'local_opts',
    'siffreader',
    'viewers',
    'use_napari'
]

class HistogramPlotter(SiffPlotter):
    """
    Extends the SiffPlotter functionality to allow
    analysis of binned arrival times of .siff files.
    Discards spatial information.

    Can be initialized with an existing SiffPlotter
    to inherit its properties

    ( e.g. hist_p = HistogramPlotter(siff_plotter)) )
    """

    def __init__(self, *args, **kwargs):
        f"""
        May be initialized from another SiffPlotter to inherit its
        attributes. Inherited attributes are:

            {inherited_params}
        """
        if not any([isinstance(arg,SiffPlotter) for arg in args]):
            # From scratch
            super().__init__(*args, **kwargs)
            self.local_opts = {
                'width' : 800,
                'colorbar' : True,
                'ylabel' : 'Number of\nphotons',
                'xlabel': 'Arrival time\n(nanoseconds)',
                'fontsize': 15,
                'toolbar' : 'above'
            }
        else:
            for arg in args:
                # Iterate until you get to the first SiffPlotter object.
                if isinstance(arg, SiffPlotter):
                    plotter = arg
                    break
            
            # inherits parameters from the provided plotter
            for param in inherited_params:
                if hasattr(plotter, param):
                    setattr(self, param, getattr(plotter, param))

        if any(map(lambda x: isinstance(x, FLIMParams)), args):
            #self.FLIMParams = []
            pass

        self.use_napari = False

    def visualize(self) -> hv.Layout:
        """ Not yet implemented """
        histograms = [self.siffreader.get_histogram(
                frames = framelist_by_color(
                    self.siffreader.im_params,
                    color-1 # 0 indexed by fcn call, 1 indexed by matlab
                )
            )
            for color in self.siffreader.im_params.colors
        ]
        raise NotImplementedError()
        return super().visualize()