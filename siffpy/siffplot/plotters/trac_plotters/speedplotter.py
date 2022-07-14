# Heading direction plots
from enum import Enum

import holoviews as hv
import numpy as np
import logging

from ...tracplotter import *
from ...utils.dims import ImageTime, AngularSpace
from ...utils.exceptions import StyleError

class SpeedPlotter(TracPlotter):
    """
    Plotter class dedicated to plotting and representing speed data from FicTrac data.

    """

    DEFAULT_LOCAL_OPTS = {}

    class SpeedStyle(Enum):
        FORWARD_SPEED    = "forward_speed"
        SIDESLIP         = "sideslip"
        ANGULAR_VELOCITY = "angular_velocity"
        SPEED            = "speed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_speed(self):
        raise NotImplementedError()

    def sideslip(self):
        raise NotImplementedError()

    def angular_velocity(self):
        raise NotImplementedError()

    def speed(self):
        raise NotImplementedError()

    def single_plot(self,
            log : LogToPlot,
            style : Union[str,SpeedStyle] = SpeedStyle.SPEED,
            **kwargs
        ) -> hv.element.path.Path:
        """
        
        Plots the wrapped heading of the fly, as reported by FicTrac as a Path element
        
        """
        if not isinstance(style, self.SpeedStyle):
            try:
                style = self.SpeedStyle(style)
            except ValueError:
                raise StyleError(self.SpeedStyle)

        raise NotImplementedError()
        # DEFAULT_OPTS = {
        #         'xlabel' : 'Time',
        #         'width'  : 1000,
        #         'line_color' : '#0071BC',
        #         'xlim' : (0.0, None),
        # }

        # t = log._dataframe['image_time']

        # wrapped_heading = log.get_dataframe_copy()['integrated_heading_lab'] # get the copy if you're modifying
        # wrapped_heading -= offset
        # wrapped_heading = wrapped_heading % (2*np.pi)
        
        # if style is self.HeadingPosition.BAR:
        #     plot_var = 2*np.pi-wrapped_heading
        #     DEFAULT_OPTS['ylabel'] = 'Bar position'
        #     DEFAULT_OPTS['yticks'] = [
        #         (offset, 'Rear'),
        #         ((np.pi + offset)%(2*np.pi), 'Front'),
        #     ]
        
        # if style is self.HeadingPosition.HEADING:
        #     plot_var = wrapped_heading
        #     DEFAULT_OPTS['ylabel'] = 'Heading'

        # if 'opts' in kwargs:
        #     OPTS_DICT = kwargs['opts']
        # else:
        #     OPTS_DICT = DEFAULT_OPTS

        # return hv.Path(
        #     split_angles_to_dict(
        #         t,
        #         plot_var,
        #         xlabel = "ImageTime",
        #         ylabel = "Angular",
        #         ),
        #     kdims = [ImageTime(), AngularSpace()] 
        # ).opts(**OPTS_DICT)
        


        


