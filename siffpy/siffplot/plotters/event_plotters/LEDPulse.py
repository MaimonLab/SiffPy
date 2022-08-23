from functools import reduce
from operator import mul

import holoviews as hv

from ..siff_plotters import EventPlotter
from ....core import SiffReader
from ....core.io.events.ledevent import LEDEvent, LEDEventType

class LEDPulsePlotter(EventPlotter):
    """
    Subclass of the EventPlotter used specifically
    to plot LED Pulses (LED On and LED Off events)
    """

    def annotate(self, element: hv.Element) -> hv.Layout:
        """
        Scans for all LED events and puts them together to
        emphasize which LEDs are on and when.
        """
        led_events = [event for event in self.siffreader.events if isinstance(event, LEDEvent)]
        if len(led_events) == 0:
            raise ValueError("No LED events recognized in provided .siff file!")
        
        led_events.sort(key = lambda x: x.frame_time) # order by frame_time first

        # behaves differently on different types of elements
        if isinstance(element, hv.Element):
            # figure out if you should overlay anything
            raise NotImplementedError()

        if isinstance(element, hv.Layout):
            # Iterate through and apply the same function
            return reduce(mul, (self.annotate(sub_element) for key, sub_element in element.data.items()) )

        raise NotImplementedError("Not yet implemented")
        #return super().annotate(element)

    @classmethod
    def qualifying(self, siffreader : SiffReader):
        return any( isinstance(event, LEDEvent) for event in siffreader.events )