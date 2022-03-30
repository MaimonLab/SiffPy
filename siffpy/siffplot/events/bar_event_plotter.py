from ...siffutils.events import BarEvent
from ..siffplotter import EventPlotter

class BarEventPlotter(EventPlotter):
    """
    A class for plotting events related to
    a bar.
    """
    def __init__(self, event : BarEvent):
        if not isinstance(event, BarEvent):
            raise ValueError("BarEventPlotter class must be initialized with a BarEvent")

            