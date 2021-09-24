# Heading direction plots

import holoviews as hv
import numpy as np

from .tracplotter import *
from ..utils.fcns import *

class HeadingPlotter(TracPlotter):

    def __init__(self, *args, **kwargs):
        super(HeadingPlotter, self).__init__(*args, **kwargs)

    @apply_opts
    def single_plot(self, log : LogToPlot, offset : float = 0, scalebar : float = None, **kwargs) -> hv.element.path.Path:
        """
        
        Plots the wrapped heading of the fly, as reported by FicTrac as a Path element
        
        """

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

        return hv.Path(
            [
                {
                    'x':t[(start+1):end],
                    'y': (wrapped_heading[(start+1):end])
                }
                for start, end in pairwise(np.where(np.abs(np.diff(wrapped_heading))>=np.pi)[0])
            ]
        ).opts(**OPTS_DICT)
        


        


