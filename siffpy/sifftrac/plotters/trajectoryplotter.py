import holoviews as hv

from .tracplotter import *

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
            
        if not kwargs is None:
            this_path.opts(**kwargs)

        return this_path
