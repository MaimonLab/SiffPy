from ...siffutils.framemetadata import FrameMetaData
from .siffevent import SiffEvent, _matlab_to_utc

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