# Heading direction plots

from typing import Type
import holoviews as hv
import numpy as np
import logging
from scipy.stats import circmean

from .tracplotter import *
from ..utils.fcns import *

class HeadingPlotter(TracPlotter):
    """
    Plotter class dedicated to plotting and representing heading from FicTrac data.

    Keyword arguments
    ----------------

    offset : float (optional)

        Pre-determined offset between some other timeseries (e.g. phase estimate of activity)
        and the heading. Subtracts offset from the heading estimate (NOT add)! So use the convention
        heading - offset = other_phase.
        
    Methods
    -------

    wrap_heading()
    """

    def __init__(self, *args, **kwargs):
        super(HeadingPlotter, self).__init__(*args, **kwargs)
        self.offset = None
        if 'offset' in kwargs:
            if type(kwargs['offset']) is float:
                self.offset = kwargs['offset']
            else:
                logging.warn(f"\nKeyword argument for offset {kwargs['offset']} is not of type float. Ignoring.")

    def wrap_heading(self, log : LogToPlot = None) -> np.ndarray:
        """
        Returns the wrapped heading with self.offset subtracted
        """
        if self.offset is None:
            offset = 0.0
        else:
            offset = self.offset
        if isinstance(log, FictracLog):
            log = LogToPlot(FLog=log)
        if log is None:
            if self.__multiple_plots():
                raise RuntimeError(
                    """
                    Current TracPlotter is storing multiple logs -- unclear which is intended.
                    Please call function again but specifying one in particular, e.g.
                    headingPlotter.wrap_heading(log)
                    """
                )
            if isinstance(self.logs[0][0], LogToPlot):
                log = self.logs[0][0]
            else:
                raise TypeError("Existing log attribute, if loaded, is not a FictracLog, LogToPlot, or list of the above.")
        if not isinstance(log, LogToPlot):
            raise TypeError(f"Argument log is not of type LogToPlot or FictracLog")
        wrapped_heading = log.get_dataframe_copy()['integrated_heading_lab'] # get the copy if you're modifying
        wrapped_heading -= offset
        return wrapped_heading % (2*np.pi)

    def fit_offset(self, phase, **kwargs) -> float:
        """
        Takes a numpy array of estimated phase (1-dimensional), computes the best aligned
        phase offset between it and heading, and stores that in this HeadingPlotter.

        Takes kwargs of FictracLog.downsample_to_imaging to determine how to align the heading to the phase data.

        Returns
        -------

        offset : float

            The phase offset between the heading and the phase estimate. This is the number that should be added to
            the phase estimate to match the heading (or, alternatively, subtracted from the heading to match phase).

        """
        downsampled_heading = self.logs[0][0].downsample_to_imaging(**kwargs)['integrated_heading_lab']
        self.offset = circmean( downsampled_heading - phase )
        return self.offset



    @apply_opts
    def single_plot(self, log : LogToPlot, offset : float = None, scalebar : float = None, **kwargs) -> hv.element.path.Path:
        """
        
        Plots the wrapped heading of the fly, as reported by FicTrac as a Path element
        
        """
        if offset is None:
            offset = 0.0
        else:
            self.offset = offset

        DEFAULT_OPTS = {
            'xlabel' : 'Time',
            'width'  : 1200,
            'line_color' : 'black',
            'yticks' : [
                (offset, 'Rear'),
                ((np.pi + offset)%(2*np.pi), 'Front'),
            ],
            'ylabel' : 'Bar position'
        }

        if offset == 0:
            DEFAULT_OPTS['yticks'].append((2*np.pi,'Rear')) # so that the top is also labeled

        t = log._dataframe['image_time']

        wrapped_heading = log.get_dataframe_copy()['integrated_heading_lab'] # get the copy if you're modifying
        wrapped_heading -= offset
        wrapped_heading = wrapped_heading % (2*np.pi)

        if 'opts' in kwargs:
            OPTS_DICT = kwargs['opts']
        else:
            OPTS_DICT = DEFAULT_OPTS

        return hv.Path(split_headings_to_dict(t, wrapped_heading)).opts(**OPTS_DICT)
        


        


