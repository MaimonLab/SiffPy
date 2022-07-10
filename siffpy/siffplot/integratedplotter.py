"""
A superclass for integrating SiffPlotter functionalities
with TracPlotter functionalities, when the two are related.

Clear examples:
    
    HeadingPlotter and PhasePlotter
    FluorescencePlotter and SpeedPlotter
"""

from abc import ABC, abstractmethod
from typing import Union

import holoviews as hv

from .siffplotter import SiffPlotter
from .tracplotter import TracPlotter
from .utils.dims import *

__all__ = [
    'IntegratedPlotter'
]

class IntegratedPlotter(ABC):
    """
    IntegratedPlotter subclasses merge the functionality of
    a SiffPlotter and a TracPlotter to produce specialized
    combination plots.
    """

    DEFAULT_SIFFPLOTTER_OPTS = {

    }

    DEFAULT_TRACPLOTTER_OPTS = {

    }

    def __init__(self, siffplotter : SiffPlotter, tracplotter : TracPlotter, **kwargs):
        f"""
        TODO: DOCSTRING!!
        """
        
        self.siffplotter = siffplotter
        self.tracplotter = tracplotter
     
    def siffplot(self, *args, **kwargs)->Union[hv.Layout, hv.Element]:
        self.siffplotter.visualize(*args, **kwargs)

    def tracplot(self, *args, **kwargs)->Union[hv.Layout, hv.Element]:
        self.tracplotter.plot(*args, **kwargs)

    @abstractmethod
    def plot(self, *args, **kwargs)->Union[hv.Layout, hv.Element]:
        raise NotImplementedError()

    @abstractmethod
    def visualize(self, *args, **kwargs)->Union[hv.Layout, hv.Element]:
        raise NotImplementedError()

        