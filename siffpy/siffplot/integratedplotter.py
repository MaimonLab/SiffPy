"""
A superclass for integrating SiffPlotter functionalities
with TracPlotter functionalities, when the two are related.

Clear examples:
    
    HeadingPlotter and PhasePlotter
    FluorescencePlotter and SpeedPlotter
"""

from abc import ABC, abstractmethod
from functools import wraps
from typing import Union

import holoviews as hv

from siffpy.siffplot.siffplotter import SiffPlotter
from siffpy.siffplot.tracplotter import TracPlotter
from siffpy.siffplot.utils.dims import *
from siffpy.siffplot.utils import apply_opts

__all__ = [
    'IntegratedPlotter'
]

class IntegratedPlotter(ABC):
    """
    IntegratedPlotter subclasses merge the functionality of
    a SiffPlotter and a TracPlotter to produce specialized
    combination plots.
    """

    DEFAULT_LOCAL_OPTS = {

    }

    DEFAULT_SIFFPLOTTER_OPTS = {

    }

    DEFAULT_TRACPLOTTER_OPTS = {

    }

    def __init__(self, siffplotter : SiffPlotter, tracplotter : TracPlotter, **kwargs):
        f"""
        TODO: DOCSTRING!!
        """
        siffplotter._local_opts = {**siffplotter._local_opts, **self.__class__.DEFAULT_SIFFPLOTTER_OPTS}
        tracplotter._local_opts = {**tracplotter._local_opts, **self.__class__.DEFAULT_TRACPLOTTER_OPTS}

        self.siffplotter = siffplotter
        self.tracplotter = tracplotter

        self._local_opts = {}

        if 'opts' in kwargs:
            self._local_opts = {**self._local_opts, **kwargs['opts']}
        else:
            self._local_opts = {**self._local_opts, **self.__class__.DEFAULT_LOCAL_OPTS}
     
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

        