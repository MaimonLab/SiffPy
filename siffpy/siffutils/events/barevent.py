from siffpy.siffutils.framemetadata import FrameMetaData
from .siffevent import SiffEvent

class BarEvent(SiffEvent):
    """
    This event is constructed when a bar is turned
    on or off.
    """
    def __init__(self, metadata : FrameMetaData):
        super().__init__(metadata)
        text = metadata.appendedText

        spl = text.split(" = ")        
        self.time_epoch = float(spl[-1])
        self.annotation = spl[0].split(" (sec since epoch)")[0]

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