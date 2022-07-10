import holoviews as hv
import numpy as np

from ...tracplotter import *

class TrajectoryPlotter(TracPlotter):

    def __init__(self, *args, **kwargs):
        super(TrajectoryPlotter, self).__init__(*args, **kwargs)

    @apply_opts
    def single_plot(self, log : LogToPlot, **kwargs)-> hv.element.path.Path:
        """ 
        
        Produces a path element of the trajectory in a single FicTracLog
        
        """

        this_path = log.dataset.to(
                    hv.Path,
                    ['integrated_position_lab_0',
                    'integrated_position_lab_1'],
                    vdims = log.tdim
                )


        square = True
        if 'square' in kwargs:
            square = kwargs['square']
        
        if square:
            bounds = (
                np.min(log.dataset['integrated_position_lab_0']),
                np.max(log.dataset['integrated_position_lab_0']),
                np.min(log.dataset['integrated_position_lab_1']),
                np.max(log.dataset['integrated_position_lab_1'])
            )
            #square_bounds = (np.min(bounds), np.max(bounds))
            this_path = this_path.opts(aspect='equal')
        if not kwargs is None:
            this_path.opts(**kwargs)

        return this_path
