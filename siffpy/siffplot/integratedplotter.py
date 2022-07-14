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

from .siffplotter import SiffPlotter
from .tracplotter import TracPlotter
from .utils.dims import *

__all__ = [
    'IntegratedPlotter'
]

def apply_opts(func):
    """
    Decorator function to apply a SiffPlotter's
    'local_opts' attribute to methods which return
    objects that might want them. Allows this object
    to supercede applied defaults, because this gets
    called with every new plot. Does nothing if local_opts
    is not defined.
    """
    @wraps(func)
    def local_opts(*args, **kwargs):
        if hasattr(args[0],'_local_opts'):
            try:
                opts = args[0]._local_opts # get the local_opts param from self
                if isinstance(opts, list):
                    return func(*args, **kwargs).opts(*opts)
                if isinstance(opts, dict):
                    return func(*args, **kwargs).opts(opts)
            except:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return local_opts

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

        