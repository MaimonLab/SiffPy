# Code for plotting trajectories
#TODO: This
from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Union
from functools import reduce
from operator import add, mul
import os, pickle

import holoviews as hv

from ..sifftrac.log_interpreter import *
from .utils import apply_opts

class TracPlotter(ABC):
    """
    
    Uses HoloViews for interactive display of FicTrac trajectories,
    and I intend to use it to interact with imaging analyses too.

    Overloads the +, +=, *, *= operators to combine trajectories (and plots).
    
    """

    DEFAULT_LOCAL_OPTS = {}

    def __init__(self, *args, opts : dict = None):
        """
        Two forms of initialization, depending on the arguments provided:

        TracPlotter(fLog : FictracLog (or list of FicTracLogs) , opts : dict = None)

            - Initializes a TracPlotter class from scratch. 

        TracPlotter(tracplotter : TracPlotter class or subclass, opts = None)

            - Initializes a TracPlotter class by inheriting the attributes from another.
            Attributes increment the references of the other (so the other TracPlotter can be
            deleted without losing the dat), but they SHARE data, so keep that in mind if you
            alter one.

        If you initialize with both, it will create a TracPlotter as if from scratch but then ADD
        the other provided TracPlotter.

        """

        ## Different types of initialization

        # If initializing using fictrac logs
        if any(map(lambda x: isinstance(x, (FictracLog, list)) , args)):
            FLog = next(filter(lambda x: isinstance(x, (FictracLog, list)) , args))
        else:
            FLog = None
        
        self._local_opts = {}
        if opts is None:
            self._local_opts = {**self._local_opts, **self.__class__.DEFAULT_LOCAL_OPTS}
        else:
            self._local_opts = opts
        
        self.figure = None
        # Now do whatever instantiation is necessary
        if not FLog is None:
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

        # Add to any type of TracPlotter provided
        if any(map(lambda x: isinstance(x, TracPlotter), args)):
            self += next(filter(lambda x: isinstance(x, TracPlotter) , args))
            return

        raise TypeError(f"Did not provide a FictracLog, a list of lists FictracLog elements, or another"
        " TracPlotter class from which to initialize.")

    ### COMBINING PLOTS FUNCTIONALITY

    @property
    def _multiple_plots(self)->bool:
        return len(self.logs)>1

    @property
    def _overlaid_plots(self)->bool:
        return any(map(lambda x: len(x) > 1,self.logs)) 

    @property
    def _multiplexed_plots(self) -> bool:
        """
        Returns true if this structure both overlays and combines plots already.
        """
        return self._multiple_plots and self._overlaid_plots

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

        if self._multiplexed_plots and other._multiplexed_plots:
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
        
        if self._multiplexed_plots and other._multiplexed_plots:
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
        if self._multiple_plots:
            self.figure = hv.Layout()
            for sublist in self.logs:
                overlay = reduce(mul, (self.single_plot(log, **kwargs) for log in sublist))
                overlay = add_scalebar(overlay, scalebar = scalebar)
                self.figure += overlay
        else:
            self.figure = reduce(mul, (self.single_plot(log, **kwargs) for log in self.logs[0]))
            self.figure = add_scalebar(self.figure, scalebar = scalebar)
        
        return self.figure

    def first_plot(self, *args, **kwargs)->hv.Element:
        if self._multiple_plots:
            return self.single_plot(self.logs[0][0])
        else:
            return self.single_plot(self.logs[0])

    @abstractmethod
    def single_plot(self, log, *args, **kwargs) -> hv.Element:
        raise NotImplementedError("This method must be implemented separately in each derived class")

    def save(self, path : str = None)->None:
        """
        Stores the object used to create these plots.

        By default, saves it next to the location of the first log.

        Arguments
        ---------
        
        path : str (optional)

            Where to save the .plotter file.
        """
        if path is None:
            path = os.path.split(self.logs[0][0].filename)[0]
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path,self.__class__.__name__)
        # Build a unique ID for this specific plotter's set of logs
        if hasattr(self.logs,'__len__'):
            for logbundle in self.logs:
                if type(logbundle )is list:
                    for log in logbundle:
                        file_name += "_" + str(self.log.filename.__hash__())
                else:
                    file_name += "_" + str(self.log.filename.__hash__())
        else:
            file_name += "_" + str(self.log.filename.__hash__())
        with open(file_name + ".plotter",'wb') as plotter_file:
            pickle.dump(self, plotter_file)
    
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
