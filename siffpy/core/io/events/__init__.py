import inspect

from siffpy.core.utils import ImParams
from siffpy.core.io.events.siffevent import SiffEvent
from siffpy.core.io.events.barevent import *
from siffpy.core.io.events.ledevent import *


EVENT_MODULES = [
    #siffpy.core.io.events.barevent,
    #siffpy.core.io.events.ledevent
]

def _isSiffEvent(obj):
    """ Returns true if event subclasses SiffEvent """
    if inspect.isclass(obj):
        return issubclass(obj, SiffEvent)
    return False

# called once
EVENT_CLASSES = [
    event_class
    for module in EVENT_MODULES
    for event_class in inspect.getmembers(module, _isSiffEvent)
]

def find_events(im_params : ImParams, metadata_list : list[FrameMetaData] = None) -> list[SiffEvent]:
    """
    Returns a list of metadata objects corresponding to all frames where
    'events' occured, i.e. in which the Appended Text field is not empty.
    """
    list_of_list_of_events = [parse_meta_as_events(meta) for meta in metadata_list if meta.hasEventTag]
    return [event for list_of_events in list_of_list_of_events for event in list_of_events]

def parse_meta_as_events(metadata : FrameMetaData) -> list[SiffEvent]:
    """
    Returns a list of all SiffEvents described in this FrameMetaData object
    """

    eventList = [
        event_class[1](metadata) # instantiate
        for event_class in EVENT_CLASSES
        if event_class[1].qualifying(metadata)  # tests if it qualifies
    ]
    return eventList