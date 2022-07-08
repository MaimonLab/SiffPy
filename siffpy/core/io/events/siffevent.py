"""
Base class of SiffEvents
"""
from abc import abstractmethod

from ..metadata import FrameMetaData
from ...timetools import SEC_TO_NANO, NANO_TO_SEC

class SiffEvent():
    """
    A SiffEvent is used to mark experiment-related
    events stored in .siff files. I felt it was a
    good idea to make a single central class (and
    family of classes) so that all components that
    might care about events could use a common language.
    """

    def __init__(self, metadata : FrameMetaData, annotation : str = None, time_epoch : int = None, frame_time : float = None):
        """
        SiffEvents must be defined from a FrameMetaData object.
        """
        if not isinstance(metadata, FrameMetaData):
            raise ValueError("SiffEvents must be initialized used a FrameMetaData")

        self.metadata = metadata

        # offset is in NANOSECONDS
        self.epoch_frame_offset : int = metadata['frameTimestamps_sec'] * SEC_TO_NANO - metadata['epoch']

        # These parameters are likely
        # derived from information contained
        # in the metadata, but not likely
        # to be metadata keys themselves
        # (e.g. time_epoch should be the time
        # of the actual event, not when it was
        # saved in the siff file)
        self.annotation : str = annotation
        self.time_epoch : int = time_epoch
        self.frame_time : float = frame_time

    def epoch_to_frame_time(self, epoch_time : int, seconds : bool = False) -> float:
        """ Converts epoch time (NANOSECONDS) to frame time (frame time in seconds).
        Optional argument seconds specifies if epoch time is ALSO in seconds... """
        if seconds:
            return (epoch_time + self.epoch_frame_offset*NANO_TO_SEC)
        else:
            return (epoch_time + self.epoch_frame_offset)*NANO_TO_SEC


    @classmethod
    @abstractmethod
    def qualifying(cls, metadata : FrameMetaData) -> bool:
        """
        Takes a FrameMetaData and determines if it contains info valid
        to produce this class
        """
        return False

    def __repr__(self):
        return "Base SiffEvent class"

def _matlab_to_utc(matlab_time : float)->int:
    """
    To fix an issue where matlab was storing time as if it were
    UTC but it was local time (and in seconds). Returns an int
    that is TRUE epoch time in UTC and in nanoseconds
    """
    if matlab_time < 1647133200: # daylight-savings switch date
        matlab_time += 3600
    return int((matlab_time + 14400) * SEC_TO_NANO)

