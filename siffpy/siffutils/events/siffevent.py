"""
Base class of SiffEvents
"""
from abc import abstractmethod

from ..framemetadata import FrameMetaData

class SiffEvent():
    """
    A SiffEvent is used to mark experiment-related
    events stored in .siff files. I felt it was a
    good idea to make a single central class (and
    family of classes) so that all components that
    might care about events could use a common language.
    """

    def __init__(self, metadata : FrameMetaData):
        """
        SiffEvents must be defined from a FrameMetaData object.
        """
        if not isinstance(metadata, FrameMetaData):
            raise ValueError("SiffEvents must be initialized used a FrameMetaData")

        self.metadata = metadata

        # These parameters are likely
        # derived from information contained
        # in the metadata, but not likely
        # to be metadata keys themselves
        # (e.g. time_epoch should be the time
        # of the actual event, not when it was
        # saved in the siff file)
        self.annotation : str = None
        self.time_epoch : float = None
        self.frame_time : float = None

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
