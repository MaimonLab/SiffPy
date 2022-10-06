from typing import Iterable, Union

import holoviews as hv
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
from bokeh.models import FixedTicker

from ...integratedplotter import IntegratedPlotter
from ..siff_plotters import PhasePlotter
from ..trac_plotters import HeadingPlotter

from ...utils.exceptions import StyleError
from ...utils.enums import CorrStyle
from ...utils.dims import ImageTime, CircCorr
from ...utils import apply_opts

from ....core import SiffReader
from ....sifftrac import FictracLog
from ....siffmath.fluorescence.traces import FluorescenceVector
from ....siffmath.phase import phase_analyses


class HeadingPhasePlotter(IntegratedPlotter):

    DEFAULT_LOCAL_OPTS = {
        'HeatMap': {
            'width' : 1000,
            #'height' : 40*len(n_sec_list),
            'clim' : (-1.0,1.0),
            'cmap' : 'coolwarm',
            'clabel' : "Circular\ncorrelation",
            'colorbar' : True,
            'colorbar_opts' : {
                    'ticker': FixedTicker(ticks=[-1, 0, 1]),
                    'title_text_font' : 'arial',
                    'title_text_font_style' : 'normal',
                    'title_text_font_size' : '12pt',
                    'major_label_text_font_size': '12pt',
                },
            #'yticks'  : [
            #    (idx, n_sec)
            #    for idx, n_sec in enumerate(n_sec_list)
            #],
            'yformatter':'$%.2f',
            'fontsize' : '9pt',
            'show_frame' : False,
            'alpha' : 1.0,
        },
        'Path': {
            'cmap': 'viridis',
            'clabel' : 'Timescale (s)',
            'width': 1000,
            'line_width': 3,
            'height' : 150,
            'show_frame': False,
            'colorbar' : True
        }
    }

    DEFAULT_SIFFPLOTTER_OPTS = {

    }

    DEFAULT_TRACPLOTTER_OPTS = {
        'Path': {
            'xlabel'        : 'Time',
            'width'         : 1000,
            'line_color'    : 'black',
            'ylabel' : 'Bar position',
            'xlim' : (0.0, None),
        }
    }

    def __init__(self, siffreader : SiffReader, fictraclog : FictracLog, fluorescence_vector : FluorescenceVector = None, **kwargs):
        
        # Type hinting for linters
        self.siffplotter : PhasePlotter
        self.tracplotter : HeadingPlotter

        pp = PhasePlotter(siffreader, fluorescence_vector = fluorescence_vector)
        hp = HeadingPlotter(fictraclog)

        super().__init__(pp, hp, **kwargs)

    def aligned_phase_plot(self,
        fluorescence_vector : FluorescenceVector = None,
        fluorescence_time   : np.ndarray         = None,
        phase_method        : str                = None,
        heading             : np.ndarray         = None,
        heading_time        : np.ndarray         = None,
        use_bar_axis        : bool               = True,
        **kwargs
        )->hv.Overlay:
        """
        Returns the overlaid phase of the fluorescence data and direction of the heading data. Does
        not actually align points, except to fit the offset. Plots the two timeseries overlaid using
        each trace's own sampling rates.
        
        Arguments
        ---------

        fluorescence_vector : FluorescenceVector or np.ndarray

            An iterable where the first axis (or the iteration axis) produces an array or
            array-like object, and across which one would estimate the phase. e.g. an
            n by time array where the phase is estimated using the "n" axis.

        fluorescence_time : np.ndarray

            A 1d array-like object whose length is the same as the "t" axis of fluorescence_vector

        phase_method : str

            Argument passed to the estimate_phase function

        heading : np.ndarray

            A 1d array-like object containing the heading to align the phase to.

        heading_time : np.ndarray

            A 1d array-like object of the same length as heading. Must be in the same
            units as fluorescence_time for alignment to be correct.

        **kwargs are passed to siffpy.siffplot.plotters.siff_plotters.phase_plotter.estimate_phase()

        Returns
        ------

        aligned_plot : hv.Overlay

            A plot with the overlaid and aligned phase and heading traces.
        """
        
        if fluorescence_vector is None:
            if self.siffplotter.data is None:
                raise ValueError("If no fluorescence data was provided on initialization, it must be provided in call to phase_plot")
            fluorescence_vector = self.siffplotter.data

        if fluorescence_time is None:
            fluorescence_time = self.siffplotter.siffreader.t_axis()

        if not (fluorescence_vector.shape[-1] == fluorescence_time.shape[-1]):
            raise ValueError("fluorescence_vector and fluorescence_time must have the same last dimension length!")

        if heading is None:
            heading = self.tracplotter.logs[0][0]._dataframe['integrated_heading_lab'].values
        
        if heading_time is None:
            heading_time = self.tracplotter.logs[0][0]._dataframe['image_time'].values

        if not (heading.shape[-1] == heading_time.shape[-1]):
            raise ValueError("heading array and heading_time array must have same last dimension length!")

        phase = self.siffplotter.estimate_phase(
            vector_timeseries = fluorescence_vector,
            phase_method = phase_method,
            **kwargs
        )

        phase_plot = self.siffplotter.plot_phase(
                fluorescence_time, 
                phase
            ).opts({'Path':{'line_color':"#698DB7"}})

        offset = phase_analyses.fit_offset(
            -heading,
            phase,
            fictrac_time = heading_time,
            phase_time = fluorescence_time
        )
        
        bar_pos_plot = self.tracplotter.plot(offset=offset, position = 'bar')

        merged_plot = phase_plot * bar_pos_plot # overlay
        
        if use_bar_axis:
            merged_plot = merged_plot.opts(
                {
                    'Path' : {
                        'yaxis' : True,
                        'yticks' : [
                            ((-offset)%(2*np.pi), 'Rear'),
                            ((np.pi - offset)%(2*np.pi), 'Front'),
                        ],
                        'ylabel' : 'Bar\nposition',
                    }
                }
            )

        return merged_plot

    @classmethod
    def align_phase_and_heading(
            self,
            phase_data      : np.ndarray,
            phase_time      : np.ndarray,
            heading_data    : np.ndarray,
            heading_time    : np.ndarray,
        )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Easy access to phase_analyses.align_two_circ_vars_timepoints.

        NOTE: NEGATES HEADING! Phase tracks BAR POSITION with Maimon lab
        convention, not heading. Returned value is NEGATIVE heading

        Downsamples to the lowest sampling frequency, circ-linearly INTERPOLATES
        the higher sampling frequency. TODO: maybe allow other downsampling methods.

        Returns
        -------
        (shared_time, aligned_phase, aligned_negative_heading) : (np.ndarray, np.ndarray, np.ndarray)
        """
        return phase_analyses.align_two_circ_vars_timepoints(
            phase_data,
            phase_time,
            -heading_data,
            heading_time
        )

    @apply_opts
    def corr_plot(self,
            shared_time     : np.ndarray,
            aligned_phase   : np.ndarray,
            aligned_heading : np.ndarray,
            windows         : Iterable[float],
            time_units      : str                  = 'sec',
            style           : Union[CorrStyle,str] = CorrStyle.HEATMAP,
            **kwargs
        ) -> hv.Element:
        """ 
        Returns a plot of the correlations between heading and the phase of the sampled data
        over the time windows provided.

        Arguments
        --------

        shared_time : np.ndarray

            Shared time axis. Not actually used for much -- the average dt is computed for determining
            the number of bins across which to take the correlation. Must be in same units as windows.

        aligned_phase : np.ndarray

            The time-aligned phase values in the shared_time timebase.

        aligned_heading : np.ndarray

            The time-aligned heading values in the shared_time timebase.

        windows : Iterable[float]

            Timescales over which to compute the sliding circular correlation. Units are same as in shared_time.
            Plot axes PRESUME this is seconds.

        time_units : str = 'sec'

            Units of time for the plot. Default is 'sec'

        style : CorrStyle or str

            What type of correlation to use. If a string is provided, must be able to be cast into
            a `siffpy.siffplot.utils.enums.CorrStyle` Enum. For more info, read the class methods
            of HeadingPhasePlotter with help(HeadingPhasePlotter).

        Returns
        -------

        plot : hv.Element

            A plot of the correlations

        """

        if isinstance(style, str):
            try:
                style = CorrStyle(style)
            except ValueError:
                raise StyleError(CorrStyle)

        if not (aligned_phase.shape == aligned_heading.shape):
            raise ValueError("Incompatible arguments -- aligned_phase and aligned_heading must have same shape")

        dt = np.mean(np.diff(shared_time))
        bin_widths = [int(x/dt) for x in windows]
        true_window_widths = [dt*x for x in bin_widths] # for labels, in units of shared_time

        correlations = phase_analyses.multiscale_circ_corr(
            aligned_phase,
            aligned_heading,
            bin_widths
        )

        if style is CorrStyle.HEATMAP:
            return self.corr_plot_heatmap(
                shared_time,
                bin_widths,
                correlations,
                true_window_widths,
                time_units,
                **kwargs
            )

        if style is CorrStyle.LINE:
            return self.corr_plot_line(
                shared_time,
                bin_widths,
                correlations,
                true_window_widths,
                time_units,
                **kwargs
            )
        
        raise ValueError("Inappropriate correlation style argument. "
            "style must either be a CorrStyle Enum or a str that can "
            "instantiate a CorrStyle Enum"
        )

    @classmethod
    def corr_plot_heatmap(
            self,
            shared_time         : np.ndarray,
            bin_widths          : Iterable[int],
            correlations        : Iterable[np.ndarray],
            true_window_widths  : Iterable[float],
            time_units          : str,
            labels              : Iterable[str]           = None,
        )->hv.Element:
        """
        Plots the input correlation traces as a heatmap. Y-axis
        is the timescale over which the circular correlation was
        taken, the x-axis is the individual time points of the 
        downsampled traces, the color axis is the value of the
        circular correlation.
        """

        heatmap_times = np.array([])
        heatmap_yval  = np.array([])
        
        # each row has a different number of samples,
        # so this seems like the easiest way : append
        # a single array onto an existing array, rather
        # than trying to do some comprehension-then-flatten
        # approach
        for idx, width in enumerate(bin_widths):    
            heatmap_times = np.append(
                heatmap_times,
                shared_time[width//2:-((width-1)//2)]
            )
            heatmap_yval = np.append(
                heatmap_yval,
                idx*np.ones((len(shared_time.flatten())-int(width)+1,))
            )
            
        correlation_flattened = np.hstack(correlations)

        return hv.HeatMap(
            (
                heatmap_times,
                heatmap_yval,
                correlation_flattened
            ),
            kdims=[ImageTime(unit=time_units), hv.Dimension("Timescale", unit=time_units)], #window_yval is unitless
            vdims=[CircCorr()],
        ).opts(
            height = 40*len(bin_widths),
            yticks = [
                    (idx, "{:.1f}".format(n_sec))
                    for idx, n_sec in enumerate(true_window_widths)
            ], 
            ylabel = "Timescale over\nwhich correlation\nis measured (sec)",
        )

    @classmethod
    def corr_plot_line(
            self,
            shared_time         : np.ndarray,
            bin_widths          : Iterable[int],
            correlations        : Iterable[np.ndarray],
            true_window_widths  : Iterable[float],
            time_units          : str,
            average_lines       : bool                  = True,
            labels              : Iterable[str]         = None,
        )->hv.Element:
        """
        Plots the input correlation traces as a heatmap. Y-axis
        is the value of the circular correlation at each time point,
        each overlaid line corresponds to a different window of time
        over which the correlation is taken.
        """

        longest_window = np.max(bin_widths) # for figuring out how many bins to average over

        # Subsamples the time_axis to omit the values over which the correlation could not be taken
        # (i.e. longer correlations requiring n bins lack values for the first n//2 and last n//2 times)
        truncated_times = []
        truncated_timescales = []
        avgd_corrs = []
        for idx, n_bins in enumerate(bin_widths):
            truncated_times.append(
                shared_time[n_bins//2:-((n_bins-1)//2)]
            )
            truncated_timescales.append(
                true_window_widths[idx]*np.ones_like(truncated_times[idx])
            )
            
            avgd_corr = correlations[idx]
            if average_lines:
                # take a rolling average to make the timescale the same as the longest trace
                window_width = int(longest_window//bin_widths[idx])
                avgd_corr = uniform_filter1d(avgd_corr, window_width)
            avgd_corrs.append(avgd_corr)

        path = hv.Path(
            [
                (time, corr, timescale)
                for time, corr, timescale in zip(truncated_times,avgd_corrs,truncated_timescales)
            ],
            kdims = [ImageTime(unit=time_units), CircCorr()],
            vdims = [hv.Dimension("Timescale", unit=time_units)],
        )
        
        return path.opts(
            {
                'Path': {
                    'color' : "Timescale",
                    'ylim' : (-1.0, 1.0),
                }
            }
        )

    def plot(self):
        raise NotImplementedError()

    def visualize(self, *args, **kwargs) -> Union[hv.Layout, hv.Element]:
        raise NotImplementedError()