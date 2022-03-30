"""
SiffPlotter class for arrival time histograms
"""

from functools import reduce
from operator import add
import logging
import random
from collections.abc import Iterable
import math

import numpy as np
import holoviews as hv

from ... import siffpy
from ...siffplot.siffplotter import SiffPlotter, apply_opts
from ...siffutils import FLIMParams
from ..utils import *
from ...siffutils.slicefcns import *

__all__ = [
    'HistogramPlotter'
]

inherited_params = [
    'local_opts',
    'siffreader'
]

class HistogramPlotter(SiffPlotter):
    """
    Extends the SiffPlotter functionality to allow
    analysis of binned arrival times of .siff files.
    Discards spatial information.

    Can be initialized with an existing SiffPlotter
    to inherit its properties

    ( e.g. hist_p = HistogramPlotter(siff_plotter)) )

    Attributes
    ----------

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

        self.FLIMParams = []
        self.FLIMParams += [param for arg in args if isinstance(arg, Iterable) for param in arg if all(arg, lambda z: isinstance(z, FLIMParams))]
        self.FLIMParams += [x for x in args if isinstance(x, FLIMParams)]

        if 'opts' in kwargs:
            self.local_opts += kwargs['opts']
        else:
            self.local_opts += [
                hv.opts.Curve({
                    'width' : 800,
                    'colorbar' : True,
                    'ylabel' : 'Number of\nphotons',
                    'xlabel': 'Arrival time\n(nanoseconds)',
                    'fontsize': 15,
                    'toolbar' : 'above',
                    'line_width' : 4
                })
            ]

    def fit(self, n_frames : int = 1000, channel : 'int|list[int]' = None, **kwargs):
        """
        Fits the arrival time distributions for the color channel(s) requested.
        Takes n_frames number of frames to perform the fit. Also accepts all kwargs
        of `siffpy.fit_exp` (for more info, refer to the `fit_exp` documentation).

        Stores the results in self.FLIMParams

        Parameters
        ----------

        n_frames : int (optional, default 1000)

            How many frames to use for each color channel to estimate the distribution of
            FLIM parameters

        channel : int or list[int]

            Color channels to fit. By default fits all color channels available. 1-indexed
            because ScanImage is MATLAB based...

        """

        if channel is None:
            channel = self.siffreader.im_params.colors
        if not isinstance(channel, list):
            channel = [channel]
        if any(color > self.siffreader.im_params.num_colors for color in channel):
            logging.warn(
                "Provided a channel number greater than number of recorded channels. " +
                f"Must be less than {self.siffreader.im_params.num_colors}."
            )
        if any(color == 0 for color in channel):
            raise ValueError("Provided channel number 0! Remember, these are 1-indexed not 0-indexed!")

        for col in channel: 
            these_frames = framelist_by_color(self.siffreader.im_params, col-1)

            sampled_number = n_frames
            if len(these_frames) < n_frames: # just in case
                sampled_number = len(these_frames)

            sampled_frames = random.sample(these_frames, sampled_number)

            channel_histogram = self.siffreader.get_histogram(
                frames = sampled_frames
            )

            # I know this function is written for multiple channels, but this way
            # I can use the fluorophore dictionary AND still have channel-wise refitting
            # happen if it's ill-behaved.
            FLIMparam = siffpy.fit_exp(channel_histogram, **kwargs)[0]
            fit_count = 0
            while ((FLIMparam.CHI_SQD > 10*n_frames) or (FLIMparam.CHI_SQD == 0)):
                if fit_count > 10:
                    logging.warn("Giving up. Using last fit.")
                    break

                logging.warn(f"Channel {col}: chi-squared / n_frames = {FLIMparam.CHI_SQD/n_frames}. " +
                    "Repeating fit with new sample."
                )
                sampled_frames = random.sample(these_frames, sampled_number)

                channel_histogram = self.siffreader.get_histogram(
                    frames = sampled_frames
                )
                FLIMparam = siffpy.fit_exp(channel_histogram, **kwargs)[0]                
                fit_count += 1

            FLIMparam.color_channel = col

            if len(self.FLIMParams) < col:
                self.FLIMParams += [[]]*(col - len(self.FLIMParams))
            
            self.FLIMParams[col-1] = FLIMparam
            
    @apply_opts
    def visualize(self, channel : 'int|list[int]' = None, text : bool = True) -> hv.Layout:
        """
        Plots arrival time histograms over the entire imaging sessions
        along with fits. If text is true, overlays text describing the fits.

        Parameters
        ----------
        channel : int or list[int]

            Color channels to fit. By default fits all color channels available.

        text : bool

            Whether to overlay text describing each histogram's fits

        Returns
        -------

        layout : hv.Layout

            A Holoviews Layout object (single column) of all color channel fits
        """

        if channel is None:
            channel = self.siffreader.im_params.colors
        if not isinstance(channel, list):
            channel = [channel]
        
        if any(color > self.siffreader.im_params.num_colors for color in channel):
            logging.warn(
                "Provided a channel number greater than number of recorded channels. " +
                f"Must be less than {self.siffreader.im_params.num_colors}."
            )
        if any(color == 0 for color in channel):
            raise ValueError("Provided channel number 0! Remember, these are 1-indexed not 0-indexed!")

        # If there are no fits provided yet, fit them all! Or the channels requested, at least.
        if not len(self.FLIMParams):
            self.fit(color=channel)        

        BIN_SIZE = self.siffreader.im_params.picoseconds_per_bin/1e3

        curveplots = []
        for col in channel:
            if col > len(self.FLIMParams):
                continue
            if not isinstance(self.FLIMParams[col-1], FLIMParams):
                continue

            # full data histograms
            histogram = self.siffreader.get_histogram(
                    frames = framelist_by_color(
                        self.siffreader.im_params,
                        col-1 # 0 indexed by fcn call, 1 indexed by matlab
                    )
                )
            
            NUM_BINS = len(histogram)
            ymax = np.max(histogram)
            bound_exp = math.ceil(math.log10(ymax))
            y_bounds = ( 10**(bound_exp-3), 10**(bound_exp) )
            # plot the data    
            this_plt = hv.Curve(
                {
                    'x': BIN_SIZE*np.arange(NUM_BINS),
                    'y':histogram
                }
            ).opts(line_color = "#000000") 
            this_plt *= hv.Curve(
                {
                    'x': BIN_SIZE*np.arange(NUM_BINS),
                    'y':np.sum(histogram)*self.FLIMParams[col-1].p_dist(
                                np.arange(0,NUM_BINS),cut_negatives=False
                        )
                }
            ).opts(line_dash='dashed')
            this_plt.opts(
                hv.opts.Curve(
                    logy=True,
                    ylim = y_bounds,
                    title = f"Channel {col}"
                )
            )

            if text:
                this_plt *= hv.Text(
                    BIN_SIZE*(0.98*NUM_BINS),
                    0.95*y_bounds[-1],
                    self.pretty_params(self.FLIMParams[col-1])
                ).opts(text_align='right', text_baseline='top' ,text_font_style = 'normal')
                pass

            curveplots.append(this_plt)
        
        final_plot = reduce(add, curveplots).opts(
            hv.opts.Curve(
                width=900, 
                xlabel='Arrival time (nanoseconds)', 
                ylabel='Photon counts'
            )
        )
        if isinstance(final_plot, hv.Layout): # if there are multiple plots
            final_plot = final_plot.cols(1)

        return final_plot
        
    def pretty_params(self, FLIMParam : FLIMParams) -> str:
        """ Converts a FLIMParams object into a nice string """

        BIN_SIZE = self.siffreader.im_params.picoseconds_per_bin/1e3

        ret_string = ""
        n_comp = FLIMParam.Ncomponents
        ret_string += f"Number of components: {n_comp}"
        for comp in range(n_comp):
            exp = FLIMParam.Exp_params[comp]
            ret_string += f"\nComponent {comp+1}: "
            ret_string += f"Tau = " + "{:.2f}".format(exp['TAU']*BIN_SIZE) + " "
            ret_string += f"Fraction = " + "{:.2f}".format(exp['FRAC']) + " "
        
        ret_string += f"\nOffset: " + "{:.2f}".format(FLIMParam.T_O * BIN_SIZE)
        ret_string += f"\nSigma : " + "{:.2f}".format(FLIMParam.IRF['PARAMS']['SIGMA'] * BIN_SIZE)
        ret_string += f"\nChi-squared : " + "{:.2f}".format(FLIMParam.CHI_SQD)
        return ret_string
