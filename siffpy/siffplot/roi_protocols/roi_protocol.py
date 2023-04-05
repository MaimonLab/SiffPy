from abc import ABC, abstractmethod
from inspect import Parameter, signature
from typing import Callable

from siffpy.siffplot.roi_protocols.rois import ROI

class ROIProtocol(ABC):
    """
    Superclass of all ROI protocols.
    
    Provides a single common interface so that
    the ROIVisualizer can call any ROI protocol
    without knowing much about it. Eliminates the
    clumsy `inspect` module usage that was
    previously required to get the arguments
    of a protocol and limited how things
    could be formatted. But these get more complicated
    and you basically have to do the type-hinting yourself...
    """

    name : str = "ROI Protocol superclass"
    base_roi_text : str = "Extract base ROI"
    on_extraction : Callable = None

    def on_click(self, segmentation_widget):
        """ Usually should be overwritten by subclass if 
        you want to implement any custom functionality.
        Should be done with a Mixin, now that I think about it """
        segmentation_widget.events.extraction_initiated()

    @abstractmethod
    def segment(self, *args, **kwargs):
        """
        Segmentation method, may be different from source
        ROI extraction
        """
        raise NotImplementedError()

    @abstractmethod
    def extract(self, *args, **kwargs)->ROI:
        """
        The main method of the ROI protocol.
        """
        raise NotImplementedError()

    @property
    def extraction_args(self):
        return {
            key : kw
            for key, kw in signature(self.extract).parameters.items()
            if kw.kind is Parameter.POSITIONAL_OR_KEYWORD
        }
    
    @property
    def segmentation_args(self):
        return {
            key : kw
            for key, kw in signature(self.return_class.segment).parameters.items()
            if kw.kind is Parameter.POSITIONAL_OR_KEYWORD
        }

    @property
    def return_class(self)->type:
        return self.extract.__annotations__["return"]
    
    @property
    def uses_reference_frames(self)->bool:
        """ Relies on extraction function having a parameter named `reference_frames`"""
        return "reference_frames" in self.extraction_args.keys()
    
    @property
    def uses_polygon_source(self)->bool:
        """ Relies on extraction function having a parameter named `polygon_source`"""
        return "polygon_source" in self.extraction_args.keys()