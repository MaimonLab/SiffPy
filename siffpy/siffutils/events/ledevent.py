from siffpy.siffutils.framemetadata import FrameMetaData
from .siffevent import SiffEvent

class LEDEvent(SiffEvent):
    """
    This event is constructed when an LED is turned
    on or off.
    """
    def __init__(self, metadata : FrameMetaData):
        super().__init__(metadata)
        text = metadata.appendedText

        notewise = text.split("\\n")

        self.brightness = None
        self.LEDOn = None

        for note in notewise:
            self.parseNote(note)

    def parseNote(self, note : str):
        """
        Parse a string and update event info accordingly
        """
        # It's the ON/OFF state
        if "time" in note:
            timestamp = note.split(" = ")
            self.annotation = timestamp[0].split(" time")[0]
            self.time_epoch = float(timestamp[-1])
        
        if "Brightness" in note:
            self.brightness = note.split(" = ")[-1].split()

        if "Lights on" in note:
            self.LEDOn = note.split(" = ")[-1].split() 

    @classmethod
    def qualifying(cls, metadata : FrameMetaData)->bool:
        try:
            if "LEDs" in metadata.appendedText:
                return True
            return False
        except:
            return False

    def __repr__(self):
        retstr = self.annotation
        retstr += f"\nAt epoch time {self.time_epoch}"
        retstr += f"\nBrightness: {self.brightness}"
        retstr += f"\nLEDs on: {self.LEDOn}"
        return retstr