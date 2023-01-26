from enum import Enum
from abc import ABC, abstractmethod, abstractproperty
from holoviews import Polygons
from numpy import ndarray

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

    @abstractmethod
    def polygons(self, slice_idx : int = None)->list[Polygons]:
        pass

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
    def get_largest_polygon(self, slice_idx : int = None, n_polygons : int = 1)->tuple[Polygons, int, int]:
        """
        Returns a tuple with:
        - the largest polygon
        - the index of the slice from which the polygon was taken
        - the index of the polygon within the slice

        Parameters
        ----------
        slice_idx : int, optional

            Index of the slice to survey. If no argument is passed,
            will survey all slices.

        n_polygons : int, optional

            Number of polygons to return. If more than one polygon,
            will return the largest, second largest, ... nth largest.

        Returns
        -------
        tuple[Polygons, int, int]
            _description_
        """
        pass

    @abstractmethod
    def get_largest_lines(self, slice_idx : int = None, n_lines : int = 2):
        pass

    @abstractmethod
    def get_largest_ellipse(self, slice_idx : int = None, n_ellipses : int = 1):
        """
        Returns a tuple with:
        - the largest ellipse
        - the index of the slice from which the ellipse was taken
        - the index of the ellipse within the slice

        Parameters
        ----------
        slice_idx : int, optional

            Index of the slice to survey. If no argument is passed,
            will survey all slices.

        n_polygons : int, optional

            Number of polygons to return. If more than one polygon,
            will return the largest, second largest, ... nth largest.

        Returns
        -------
        tuple[Polygons, int, int]
            _description_
        """
        pass

    @abstractmethod
    def source_image(self, slice_idx : int = None)->ndarray:
        pass

    @abstractproperty
    def orientation(self):
        pass