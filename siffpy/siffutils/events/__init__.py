
from siffpy.siffutils.framemetadata import FrameMetaData

import inspect

from .siffevent import SiffEvent
from .barevent import *
from .ledevent import *


EVENT_MODULES = [
    barevent,
    ledevent
]

def parseMetaAsEvents(metadata : FrameMetaData) -> list[SiffEvent]:
    """
    Returns a list of all SiffEvents described in this FrameMetaData object
    """

    def isSiffEvent(obj):
        if inspect.isclass(obj):
            return issubclass(obj, SiffEvent)
        return False

    eventList = [
        event_class[1](metadata) # instantiate
        for module in EVENT_MODULES
        for event_class in inspect.getmembers(module, isSiffEvent)
        if event_class[1].qualifying(metadata)  # tests if it qualifies
    ]
    return eventList