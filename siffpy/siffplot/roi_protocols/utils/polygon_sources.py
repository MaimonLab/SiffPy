from enum import Enum
from abc import ABC, abstractmethod, abstractproperty

class VizBackend(Enum):
    NAPARI = 'Napari'
    HOLOVIEWS = 'Holoviews'

class PolygonSource(ABC):
    """
    Backend-invariant polygon source so that
    a single function can be used with either
    napari or holoviews. Each subclass should
    implement each method set up here, and 
    almost every method should take a `PolygonSource`
    as an argument, rather than a backend-specific
    object!
    """

    
    def __init__(self, interface : VizBackend, source : object):
        self.interface = interface
        self.source = source

    @abstractproperty
    def polygons(self):
        if self.interface == VizBackend.NAPARI:
            return
        if self.interface == VizBackend.HOLOVIEWS:
            return

    @abstractmethod
    def to_napari(self):
        if self.interface == VizBackend.NAPARI:
            return
        if self.interface == VizBackend.HOLOVIEWS:
            raise NotImplementedError("Conversion not yet implemented")

    @abstractmethod
    def to_holoviews(self):
        if self.interface == VizBackend.NAPARI:
            raise NotImplementedError("Conversion not yet implemented")
        if self.interface == VizBackend.HOLOVIEWS:
            return

    @abstractmethod
    def get_largest_polygon(self, slice_idx = None, n_polygons = 1):
        pass

    @abstractmethod
    def get_largest_lines(self, slice_idx = None, n_lines = 2):
        pass

    @abstractmethod
    def get_largest_ellipse(self, slice_idx = None, n_ellipses = 1):
        pass
