"""
Generic EventPlotter.

Can be overwritten for different types of events

"""
from functools import reduce
from operator import mul
from abc import abstractmethod

import holoviews as hv

from ....core import SiffReader
from ....core.io import SiffEvent
from ...siffplotter import SiffPlotter
from ...utils import apply_opts
from ...utils.dims import *


__all__ = [
    'EventPlotter'
]

class EventPlotter(SiffPlotter):
    """
    Extends the SiffPlotter functionality to allow
    analysis of events in a .siff file.

    Can be initialized with an existing SiffPlotter
    to inherit its properties

    ( e.g. ep = EventPlotter(siff_plotter)) )

    The visualize method returns a HoloViews Overlay object
    with all events stored by the SiffReader annotated.
    """

    DEFAULT_OPTS = {
        'Arrow' : {
            'width' : 1000,
            'show_frame' : False,
            'yaxis' : None,
            'xaxis' : None,
            'height' : 50,
            'padding' : 0,
            'border' : 0,
        },
        'Text' : {
            'width' : 1000,
            'fontsize' : 15,
            'show_frame' : False,
            'yaxis' : None,
            'xaxis' : None,
            'height' : 50,
            'padding' : 0,
            'border' : 0,
        },
        'Overlay' : {
            'width' : 1000,
            'show_frame' : False,
            'yaxis' : None,
            'xaxis' : None,
            'height' : 50,
            'padding' : 0,
            'border' : 0
        }
    }

    INHERITED_PARAMS = [
        'local_opts',
        'siffreader',
        'reference_frames',
    ]
        
    @apply_opts
    def plot_event(self, event : SiffEvent, direction = 'v')->hv.Overlay:
        """
        Generic event annotation.

        Returns a small Overlay object with an arrow demarcating the time of an event and text annotation
        """
        arrow = hv.element.Arrow(event.frame_time, 0, direction=direction, kdims=[ImageTime(), AnnotationAxis()]).opts(ylim=(0,10))
        text = hv.element.Text(event.frame_time, 2.5, event.annotation, valign='bottom', kdims=[ImageTime(), AnnotationAxis()]).opts(ylim=(0,10))
        return arrow*text

    def annotation_element(self)->hv.Layout:
        """ Returns a single element for all the events together """
        if self.siffreader.events is None:
            return None
        if len(self.siffreader.events) == 0:
            return None
        return reduce(mul, (self.plot_event(event) for event in self.siffreader.events))

    def annotate(self, element : hv.Element)->hv.Layout:
        """ Returns a HoloViews Layout object that has been annotated with the EventPlotter """
        annotations = self.annotation_element() 
        #if len(element.kdims): # no idea what this was meant to do, leaving it in in case I remember.
            #annotations = annotations
        if annotations is None:
            return element
        return (
            annotations + element
        ).opts(transpose=True)

    @apply_opts
    def visualize(self, *args, **kwargs)->hv.Layout:
        """
        If timeseries has only one non-singleton dimension, returns
        a HoloViews.Curve element.

        Otherwise, returns a HoloViews Layout object that shows a heatmap visualization of the changes in fluorescence
        in the associated timeseries.

        If None is provided, then it will check if the current ROI has been segmented.
        If so, it will return a HoloViews Layout with the default analyses for vector_timeseries
        performed. If not, it will return a HoloViews Curve!

        Stores the timeseries returned, either in the vector_timseries or roi_timeseries attribute.

        Currently HoloViews only.
        """
        raise NotImplementedError()

    @classmethod
    def qualifying(cls, siffreader : SiffReader) -> bool:
        """
        Takes a SiffReader and determines if it contains info valid
        to produce this class. To be implemented by each custom EventPlotter
        to automate detection and annotation of event types.
        """
        return False