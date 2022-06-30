import inspect

from ...utils import ImParams
from .siffevent import SiffEvent
from .barevent import *
from .ledevent import *


EVENT_MODULES = [
    barevent,
    ledevent
]

def find_events(im_params : ImParams, metadata_list : list[FrameMetaData] = None) -> list[SiffEvent]:
    """
    Returns a list of metadata objects corresponding to all frames where
    'events' occured, i.e. in which the Appended Text field is not empty.
    """
    list_of_list_of_events = [parseMetaAsEvents(meta) for meta in metadata_list if meta.hasEventTag]
    return [event for list_of_events in list_of_list_of_events for event in list_of_events]

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