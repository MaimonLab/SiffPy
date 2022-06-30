from enum import Enum

from ..metadata import FrameMetaData
from .siffevent import SiffEvent, _matlab_to_utc


class BarEventType(Enum):
    """
    There are types of Bar Events.
    At present, they're On/Off, but
    there could in principle be many others.

    Would love to use a StrEnum but those are new in 3.11
    and I'm not ready to impose that level of Python
    versioning on users because I don't think some of the
    dependencies are up to date on that.
    """
    ON_EVENT = 'on'
    OFF_EVENT = 'off'
    UNDEFINED = 'undefined'

class BarEvent(SiffEvent):
    """
    This event is constructed when a bar is turned
    on or off.
    """
    def __init__(self, metadata : FrameMetaData):
        super().__init__(metadata)
        text = metadata.appendedText

        spl = text.split(" = ")        
        
        if "(sec" in spl[0]: # OLD MATLAB ISSUE
            self.time_epoch = _matlab_to_utc(float(spl[-1]))
        else:    
            self.time_epoch = int(spl[-1])
        self.frame_time = float(self.epoch_to_frame_time(self.time_epoch))
        self.annotation = spl[0].split(" (")[0]
        self.event_type : BarEventType = BarEventType.UNDEFINED

        if "Bar on" in self.annotation:
            self.event_type = BarEventType.ON_EVENT
        if "Bar off" in self.annotation:
            self.event_type = BarEventType.OFF_EVENT

    @classmethod
    def qualifying(cls, metadata : FrameMetaData)->bool:
        try:
            if "Bar" in metadata.appendedText:
                return True
            return False
        except:
            return False

    def __repr__(self):
        retstr = self.annotation
        retstr += f"\nAt epoch time {self.time_epoch}"
        return retstr