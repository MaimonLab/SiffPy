"""
Specialized Dimension subclasses that exist for common use
cases so that their axes can be shared across HoloViews elements
in a Layout without worrying about breaking code that combines
elements when I adjust one.
"""

import holoviews as hv
import numpy as np

class FluorescenceAxis(hv.Dimension):
    """
    A Dimension class for shared Fluorescence outputs.
    If no info is provided, presumes that the relevant
    label is 'dF/F'.
    """
    def __init__(self, method : str = None, **kwargs):
        """ hv.Dimension('FluorescenceAxis', method) """
        if not isinstance(method, str):
            method = "dF/F" # default
        super().__init__(("FluorescenceAxis", method), **kwargs)

class FlimAxis(hv.Dimension):
    def __init__(self, method : str = None, **kwargs):
        """ hv.Dimension('Flim', method) """
        if not isinstance(method, str):
            method = 'Empirical lifetime'
        super().__init__(("Flim", method), **kwargs)

class ArrivalTime(hv.Dimension):
    """
    A dimension class for plotting photon arrival times
    """
    def __init__(self, **kwargs):
        """ hv.Dimension('ArrivalTime', 'Photon arrival time', unit='nanoseconds') """
        if not "unit" in kwargs:
            kwargs["unit"] = "nanoseconds"
        super().__init__(("ArrivalTime", "Photon arrival time"), **kwargs)

class HistogramCounts(hv.Dimension):
    """
    A dimension class for plotting photon histogram bins
    """
    def __init__(self, **kwargs):
        """ "hv.Dimension('HistogramCounts', 'Photons') """
        super().__init__(("HistogramCounts", "Photons"), **kwargs)

class ImageTime(hv.Dimension):
    """
    A Dimension class that tracks
    "ImageTime". Is almost always
    an x-axis for plots of information
    that varies across an imaging experiment.
    """
    def __init__(self, **kwargs):
        """ hv.Dimension('ImageTime', 'Time', unit='sec') """
        if not "unit" in kwargs:
            kwargs["unit"] = "sec"
        super().__init__(("ImageTime", "Time"), **kwargs)
        

class AngularSpace(hv.Dimension):
    """
    A dimension class that tracks the angular coordinates of,
    for example, ROIs and fluorescence traces, and their corresponding
    metadata, or a phase, or a bar's angular coordinate.
    """

    def __init__(self, **kwargs):
        """ hv.Dimension('Angular', 'Angular\ncoordinate', unit='radians', range=(0, 2*np.pi))"""
        if not "unit" in kwargs:
            kwargs["unit"] = "radians"
        if not "range" in kwargs:
            kwargs["range"] = (0, 2*np.pi)
        super().__init__(("Angular", "Angular\ncoordinate"), **kwargs)

class CircCorr(hv.Dimension):
    """
    A dimension class that tracks the circular correlation between
    two circular variables
    """
    def __init__(self, **kwargs):
        """ hv.Dimension('CircCorr', 'Circular correlation', range=(-1.0,1.0)) """
        if not "range" in kwargs:
            kwargs["range"] = (0, 2*np.pi)
        super().__init__(("CircCorr","Circular correlation"), **kwargs)

class UnwrappedHeading(hv.Dimension):
    """
    A dimension class for unbounded angles -- units are REVOLUTIONS
    not radians by default.
    """

    def __init__(self, label : str = "Unwrapped heading", **kwargs):
        """ hv.Dimension('UnwrappedAngle', label, unit='revolutions') """
        if not "unit" in kwargs:
            kwargs["unit"] = "revolutions"
        super().__init__(("UnwrappedAngle", label), **kwargs)


class AnnotationAxis(hv.Dimension):
    """
    A shared dimension for annotations on Events
    """

    def __init__(self, **kwargs):
        """ hv.Dimension('Annotation', " ", unit=None, range=(0,10))"""
        if "unit" in kwargs:
            kwargs["unit"] = None
        if not "range" in kwargs:
            kwargs["range"] = (0,10)
        super().__init__(("Annotation", " "), **kwargs)
