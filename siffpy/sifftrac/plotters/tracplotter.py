# Code for plotting trajectories
#TODO: This
from __future__ import annotations
from typing import Union

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

from numpy import add
from ..log_interpreter.fictraclog import FictracLog
from ..log_interpreter.logtoplot import LogToPlot


def apply_opts(func):
    """
    Decorator function to apply a TracPlotter's
    'local_opts' attribute to methods which return
    objects that might want them. Allows this object
    to supercede applied defaults, because this gets
    called with every new plot.
    """
    def local_opts(*args, **kwargs):
        if hasattr(args[0],'local_opts'):
            try:
                opts = args[0].local_opts # get the local_opts param from self
                return func(*args, **kwargs).opts(opts)
            except Exception as e:
                raise RuntimeError(f"Error applying local opts!:\n{e}")
        else:
            return func(*args,**kwargs)
    return local_opts

class TracPlotter():
    """
    
    Uses HoloViews for interactive display of FicTrac trajectories,
    and I intend to use it to interact with imaging analyses too.

    Overloads the +, +=, *, *= operators to combine trajectories (and plots).
    
    """

    def __init__(self, FLog : Union[FictracLog,list[FictracLog], list[list[FictracLog]]], opts : dict = None):
        
        self.figure = None
        if not opts is None:
            self.local_opts = opts

        if isinstance(FLog, FictracLog):
            self.logs = [[LogToPlot(FictracLog=FLog)]]
            return
        
        if isinstance(FLog, list):
            # If it's a list of FictracLogs, they're intended to be overlaid
            if all(map(lambda x: isinstance(x, FictracLog), FLog)):
                self.logs = [[LogToPlot(FictracLog = flog) for flog in FLog]]
                return
            
            # If it's a list of lists of FictracLogs, then they're supposed
            # to be plotted separately
            if all(map(lambda y:
                all(map(lambda x: isinstance(x, FictracLog),y)),
                FLog
            )):
                self.logs = [
                    [   
                        LogToPlot(FictracLog = flog)
                        for flog in shared_logs
                    ]   for shared_logs in FLog
                ]
                return

        raise TypeError(f"Argument FLog is not of type FictracLog or a list of lists FictracLog elements")

    ### COMBINING PLOTS FUNCTIONALITY

    def __multiple_plots(self)->bool:
        return len(self.logs)>1

    def __overlaid_plots(self)->bool:
        return any(map(lambda x: len(x) > 1,self.logs)) 

    def __multiplexed_plots(self) -> bool:
        """
        Returns true if this structure both overlays and combines plots already.
        """
        return self.__multiple_plots() and self.__overlaid_plots()

    def __add__(self, other : TracPlotter)-> TracPlotter:
        """
        Overloaded to combine plots as separate subplots and append two lists of trajectories
        """
        if not isinstance(other, TracPlotter):
            return NotImplemented

        NewTracPlotter = TracPlotter(FLog = self.logs + other.logs)       

        # If either of them has a figure, the new TracPlot should too
        if not(self.figure is None and other.figure is None):
            # Combine their figures Holoviews style, so that this figure inherits modifications made to the others
            if self.figure is None:
                NewTracPlotter.figure = other.figure
            elif other.figure is None:
                NewTracPlotter.figure = self.figure
            else:
                NewTracPlotter.figure = self.figure + other.figure
        
        return NewTracPlotter
        
    def __iadd__(self, other : TracPlotter)->TracPlotter:
        if not isinstance(other, TracPlotter):
            return NotImplemented

        self.logs.extend(other.logs)
        
        if not self.figure is None:
            # If we have a figure, we should update it (in-place)
            if other.figure is None:
                # If the other TracPlotter has no figure but this one does, we should make it!
                # TODO: Make this inherit the in-place plot modifications -- why else combine in place?
                self.figure += other.plot()
            else:
                self.figure += other.figure
        
        return self


    def __imul__(self, other: TracPlotter)->TracPlotter:
        if not isinstance(other, TracPlotter):
            return NotImplemented

        if self.__multiplexed_plots() and other.__multiplexed_plots():
            raise TypeError("Operator *= is unsupported for two TracPlotters with multiplexed figures " +
            "(i.e. figures that both overlay and compose FictracLog plots)")
        
        if isinstance(self.figure,hv.core.layout.Layout) and isinstance(other.figure, hv.core.layout.Layout):
            raise TypeError("Unsupported operator *= for two TracPlotters with Layout-type figures")

        # Take the (potentially overlay) element in the 1-element log list and add it to all the longer one
        if len(self.logs) == 1:
            self.logs = list(map(lambda x: x + self.logs[0], other.logs))
        else:
            self.logs = list(map(lambda x: x + other.logs[0], self.logs))

        if not self.figure is None:
            # If we have a figure, we should update it (in-place)
            if other.figure is None:
                # If the other TracPlotter has no figure but this one does, we should make it!
                # TODO: Make this inherit the in-place plot modifications -- why else combine in place?
                self.figure *= other.plot()
            else:
                self.figure *= other.figure
        return self
        

    def __mul__(self, other : TracPlotter)->TracPlotter:
        """
        Overloaded to overlay plots and pool trajectories in a list
        """
        if not isinstance(other, TracPlotter):
            return NotImplemented
        
        if self.__multiplexed_plots() and other.__multiplexed_plots():
            raise TypeError("Operator * is unsupported for two TracPlotters with multiplexed figures " +
            "(i.e. figures that both overlay and compose FictracLog plots)")
        
        if isinstance(self.figure,hv.core.layout.Layout) and isinstance(other.figure, hv.core.layout.Layout):
            raise TypeError("Unsupported operator * for two TracPlotters with Layout-type figures")
        
        # Take the (potentially overlay) element in the 1-element log list and add it to all the longer one
        if len(self.logs) == 1:
            f_list = list(map(lambda x: x + self.logs[0], other.logs))
        else:
            f_list = list(map(lambda x: x + other.logs[0], self.logs))
        NewTracPlotter = TracPlotter(FLog = f_list)

        # If either of them has a figure, the new TracPlotter should too
        if not(self.figure is None and other.figure is None):
            # Combine their figures Holoviews style, so that this figure inherits modifications made to the others
            if self.figure is None:
                NewTracPlotter.figure = other.figure
            elif other.figure is None:
                NewTracPlotter.figure = self.figure
            else:
                NewTracPlotter.figure = self.figure * other.figure
        
        return NewTracPlotter

    def __rmul__(self, other : TracPlotter)->TracPlotter:
        """
        Shouldn't ever happen except when multiplied by a non TracPlotter, so we'll always return NotImplemented.

        May need to be adjusted.
        """
        if not isinstance(other, TracPlotter):
            return NotImplemented
        raise NotImplementedError()

    ### PLOTTING FUNCTIONALITY

    @apply_opts
    def plot(self, scalebar : float = None, **kwargs):
        """

        Produce a Holoviews Path object for the fly's trajectory,
        customizable as seen fit.

        Accepts Holoviews opts kwargs as keyword arguments.        

        PARAMETERS
        ----------

        scalebar : float

            Size of the scalebar, in mm (if not None)

        RETURNS
        -------

        figure : hv.Layout or hv.Overlay

        """
        if self._TracPlotter__multiple_plots():
            self.figure = hv.Layout()
            for sublist in self.logs:
                overlay = hv.Overlay()
                for log in sublist:
                    overlay *= self.single_plot(log, **kwargs)
                overlay = add_scalebar(overlay, scalebar = scalebar)
                self.figure += overlay
        else:
            self.figure = hv.Overlay()
            for log in self.logs[0]:
                self.figure *= self.single_plot(log, **kwargs)
            self.figure = add_scalebar(self.figure, scalebar = scalebar)
        
        return self.figure

    def single_plot(self, log, *args, **kwargs) -> hv.Element:
        raise NotImplementedError("This method must be implemented separately in each derived class")

    
## LOCAL SHARED FCNS
def add_scalebar(fig : Union[hv.Layout, hv.Overlay], scalebar : float = None):
    if not scalebar is None:
        scalebar = float(scalebar)
        xcord, ycord = fig.range('X')[-1], fig.range('Y')[0] # bottom right
        #add the scalebar
        fig *=  hv.Curve(
            [(xcord - scalebar,0.95*ycord), (xcord, 0.95*ycord)]
            ).opts(
                color=(0,0,0) # black
            )
        #add the text
        fig *= hv.Text(xcord-scalebar, 0.9*ycord, f"{scalebar} millimeters")
    return fig
