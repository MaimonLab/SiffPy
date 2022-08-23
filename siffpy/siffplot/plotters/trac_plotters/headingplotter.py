# Heading direction plots
from enum import Enum

import holoviews as hv
import numpy as np
import logging


from ....core.utils.circle_fcns import split_angles_to_dict
from ....sifftrac.log_interpreter import _ORIGINAL_FICTRAC_ROS_ZERO_HEADING
from ...tracplotter import *
from ...utils.dims import ImageTime, AngularSpace
from ...utils.exceptions import StyleError

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

    DEFAULT_LOCAL_OPTS = {
        'xlabel' : 'Time',
        'width'  : 1000,
        'line_color' : 'black',
        'xlim' : (0.0, None),
    }

    class HeadingPosition(Enum):
        BAR     = 'bar'
        HEADING = 'heading'

    def __init__(self, *args, **kwargs):
        super(HeadingPlotter, self).__init__(*args, **kwargs)
        self.offset = None
        if 'offset' in kwargs:
            if type(kwargs['offset']) is float:
                self.offset = kwargs['offset']
            else:
                logging.warn(f"\nKeyword argument for offset {kwargs['offset']} is not of type float. Ignoring.")

    def wrapped_heading(self, log : LogToPlot = None) -> np.ndarray:
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
            if self._multiple_plots:
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
        wrapped_heading = np.exp(log.get_dataframe_copy()['integrated_heading_lab']*1j) # get the copy if you're modifying
        wrapped_heading *= np.exp(offset*1j)
        return np.angle(wrapped_heading)+np.pi

    def fit_offset(self, phase : np.ndarray, **kwargs) -> float:
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
        raise NotImplementedError()
        return self.offset

    def single_plot(self,
            log : LogToPlot,
            offset : float = None,
            scalebar : float = None,
            style : Union[str,HeadingPosition] = HeadingPosition.BAR,
            **kwargs
        ) -> hv.element.path.Path:
        """
        
        Plots the wrapped heading of the fly, as reported by FicTrac as a Path element
        
        """
        if offset is None:
            offset = 0.0
        else:
            self.offset = offset

        if not isinstance(style, self.HeadingPosition):
            try:
                style = self.HeadingPosition(style)
            except ValueError:
                raise StyleError(self.HeadingPosition)


        if log._OLD_PROJECTOR_DRIVER: # back compatibility
            offset += _ORIGINAL_FICTRAC_ROS_ZERO_HEADING

        DEFAULT_OPTS = {
                'xlabel' : 'Time',
                'width'  : 1000,
                'line_color' : 'black',
                'xlim' : (0.0, None),
        }

        if offset == 0:
            DEFAULT_OPTS['yticks'].append((2*np.pi,'Rear')) # so that the top is also labeled

        t = log._dataframe['image_time']

        wrapped_heading = log.get_dataframe_copy()['integrated_heading_lab'] # get the copy if you're modifying
        wrapped_heading -= offset
        wrapped_heading = wrapped_heading % (2*np.pi)
        
        if style is self.HeadingPosition.BAR:
            plot_var = 2*np.pi-wrapped_heading
            DEFAULT_OPTS['ylabel'] = 'Bar position'
            DEFAULT_OPTS['yticks'] = [
                (offset, 'Rear'),
                ((np.pi + offset)%(2*np.pi), 'Front'),
            ]
        
        if style is self.HeadingPosition.HEADING:
            plot_var = wrapped_heading
            DEFAULT_OPTS['ylabel'] = 'Heading'

        if 'opts' in kwargs:
            OPTS_DICT = kwargs['opts']
        else:
            OPTS_DICT = DEFAULT_OPTS

        return hv.Path(
            split_angles_to_dict(
                t,
                plot_var,
                xlabel = "ImageTime",
                ylabel = "Angular",
                ),
            kdims = [ImageTime(), AngularSpace()] 
        ).opts(**OPTS_DICT)
        


        


