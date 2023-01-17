# Heading direction plots
from enum import Enum

import holoviews as hv
import numpy as np
import logging

from siffpy.siffplot.tracplotter import *
from siffpy.siffplot.utils.dims import AngularVelocityAxis, ImageTime, SpeedAxis
from siffpy.siffplot.utils.exceptions import StyleError
from siffpy.core.timetools import rolling_avg
from siffpy.core.utils.circle_fcns import circ_diff

class SpeedPlotter(TracPlotter):
    """
    Plotter class dedicated to plotting and representing speed data from FicTrac data.

    """

    DEFAULT_LOCAL_OPTS = {
        'width'  : 1000,
        'height' : 150,
        'line_color' : '#0071BC',
        'xlim' : (0.0, None),
    }

    class SpeedStyle(Enum):
        FORWARD_SPEED    = "forward_speed"
        SIDESLIP         = "sideslip"
        ANGULAR_VELOCITY = "angular_velocity"
        SPEED            = "speed"
        ALL              = "all"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def single_plot(self,
            log : LogToPlot,
            style : Union[str,SpeedStyle] = SpeedStyle.ALL,
            rolling_avg : float = 1.0, # units are SECONDS
            **kwargs
        ) -> hv.element.path.Path:
        """
        
        Plots the wrapped heading of the fly, as reported by FicTrac as a Path element

        Arguments
        ---------

        log : LogToPlot

            Annotated FictracLog with image_time coordinates

        style : str | speedplotter.SpeedStyle

            SpeedStyle enum or a string that can be initialized as such, defined as a class
            attribute of the SpeedPlotter. Options can be identified with help(SpeedPlotter.SpeedStyle).

        rolling_avg : float

            Timescale over which to take a rolling average. Default is 1.0 seconds. Units are presumed to
            be seconds (but in practice are whatever units the timebase of the argument `log` are!).

        Returns
        -------

        speed_plot : hv.Path | hv.Layout

            Returns an hv.Path object, unless SpeedStyle.ALL is passed for the style, in which case it
            returns a Layout with all of the plots stacked vertically.

        
        """
        if not isinstance(style, self.SpeedStyle):
            try:
                style = self.SpeedStyle(style)
            except ValueError:
                raise StyleError(self.SpeedStyle)

        # Handled separately.
        if style is self.SpeedStyle.ALL:
            return (
                self.single_plot(log, style = 'forward_speed', rolling_avg=rolling_avg) +
                self.single_plot(log, style ='sideslip', rolling_avg=rolling_avg) + 
                self.single_plot(log, style = 'angular_velocity', rolling_avg=rolling_avg) +
                self.single_plot(log, style = 'speed', rolling_avg=rolling_avg)
            ).cols(1) # vertical overlay

        if style is self.SpeedStyle.FORWARD_SPEED:
            (t_axis, speed) = forward_speed_from_log(log, rolling_avg)
            label = "Forward speed\n"
            unit = 'mm/sec'
        if style is self.SpeedStyle.SIDESLIP:
            (t_axis, speed) = sideslip_from_log(log, rolling_avg)
            label = "Sideslip\n"
            unit = "mm/sec"
        if style is self.SpeedStyle.ANGULAR_VELOCITY:
            (t_axis, speed) = angular_velocity_from_log(log, rolling_avg)
            yaxis = AngularVelocityAxis()
        if style is self.SpeedStyle.SPEED:
            (t_axis, speed) = speed_from_log(log, rolling_avg)
            label = "Movement speed\n"
            unit = "mm/sec"
            yaxis = SpeedAxis(label, unit=unit)

        return hv.Path(
            (t_axis, speed),
            kdims = [
                ImageTime(),
                yaxis
            ]
        )
        
def forward_speed_from_log(log, window_width : float)->tuple:
    raise NotImplementedError()

def sideslip_from_log(log, window_width : float)->tuple:
    raise NotImplementedError()

def angular_velocity_from_log(log, window_width : float)->tuple:
    if not 'image_time' in log._dataframe:
        logging.warning("No image time axis! Using timestamps instead!")
        time_axis = log._dataframe['timestamps'].values
    else:
        time_axis = log._dataframe['image_time'].values
    heading = log._dataframe['integrated_heading_lab'].values
    v_axis = circ_diff(heading)
    v_axis = v_axis*180.0/np.pi # degrees
    delta_t = np.mean(np.diff(time_axis))
    time_axis = time_axis[:-1]

    return (time_axis, rolling_avg(v_axis, time_axis, window_width)/delta_t)

def speed_from_log(log, window_width : float)->tuple:
    if not 'image_time' in log._dataframe:
        logging.warning("No image time axis! Using timestamps instead!")
        time_axis = log._dataframe['timestamps'].values
    else:
        time_axis = log._dataframe['image_time'].values

    v_axis = log._dataframe['animal_movement_speed'].values
    delta_t = np.mean(np.diff(time_axis))

    return (time_axis, rolling_avg(v_axis,time_axis,window_width)/delta_t)
    


