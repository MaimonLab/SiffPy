class FrameMetaData(object):
    """
    A glorified dictionary, contains the metadata stored on a framewise
    basis in .siff files (and, often, .tiff files). Can be treated and
    accessed like a dict (i.e. with set and get items like metaData['newKey'] = x)
    or like a class with attributes (e.g. metaData.newKey = x).

    Parameters
    ----------

    frameNumbers : int (or list of ints if an averaged frame)

        The number (into the acquisition) of the corresponding frame

    frameTimestamps_sec : float (or list of floats if an averaged frame)

        The time since the beginning of the acquisition of the frame trigger,
        i.e. the event at the *START* of the frame (in SECONDS!)

    epoch : int (or list of ints if an averaged frame)

        The time in EPOCH nanoseconds of the frame trigger (the *START* of the frame).
        Accuracy is constrained by 1) the Windows scheduler, and 2) the synchronization
        of the ScanImage clock to any other clocks you may be also measuring events with
        in EPOCH time. I would be skeptical of presuming accuracy better than 5 milliseconds,
        even if the measurements look more regular than that.

    appendedText : str

        A string written to the frame if some sort of event happened in ScanImage that
        the user wanted to log in the .siff file.
    
    endOfAcquisition : bool

        Whether this is the end of a single 'Acquisition' in ScanImage (i.e. a round of collected frames).
        Some imaging paradigms will use the 'Loop' feature, in which case this may occur before the end
        of the actual experiment

    endOfAcquisitionMode : bool

        Whether this is the end of the 'Loop' or 'Grab'. Is not true until the final Acquisition if
        Looping.

    dcOverVoltage : bool

        A parameter stored by vanilla ScanImage that I neither use nor save at time of this writing
        (Dec. 27, 2021)

    acqTriggerTimestamps_sec : float

        A parameter stored by vanilla ScanImage that will maybe one day store the timestamps
        of triggers going in and out of the vDAQ if I can ever get access to an API that shares
        that information with me. Proprietary silliness...

    nextFileMarkerTimestamps_sec : float

        A parameter stored by vanilla ScanImage that maybe I will use if I can ever figure
        out what it was supposed to be.
    """

    CORE_META_PARAMS = (
        "frameNumbers",
        "frameTimestamps_sec"
    )

    OPTIONAL_META_PARAMS = (
        "frameNumberAcquisition",
        "acqTriggerTimestamps_sec",
        "nextFileMarkerTimestamps_sec",
        'endOfAcquisition',
        'endOfAcquisitionMode',
        'dcOverVoltage',
        'epoch'
    )

    EVENT_TAGS = ( # tags that could be considered events
        "Appended text",
    )

    def __init__(self, param_dict : dict):
        """
        Initialized by reading in a param dict extracted by the C API.
        """
        self.hasEventTag = False

        for key in FrameMetaData.CORE_META_PARAMS:
            if not (key in param_dict) or (key.lower() in param_dict):
                raise KeyError(f"Input parameter dictionary is incomplete. Lacks {key} (and possibly others)")
            setattr(self, key, param_dict[key])
        
        for key in FrameMetaData.OPTIONAL_META_PARAMS:
            if (key in param_dict) or (key.lower() in param_dict):
                setattr(self, key, param_dict[key])

        for key in FrameMetaData.EVENT_TAGS:
            if (key in param_dict) or (key.lower() in param_dict):
                setattr(self,'hasEventTag', True)
                if key == "Appended text":
                    setattr(self,'appendedText', param_dict[key])
                else:
                    setattr(self, key, param_dict[key])

    def __getitem__(self, key : str) -> None:

        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Metadata field {key} does not exist for this frame")

    def __setitem__(self, key : str, value) -> None:
        setattr(self, key, value)

    def items(self):
        return [(attr_key, getattr(self,attr_key)) for attr_key in self.__dict__.keys()]
    
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return [getattr(self, key) for key in self.__dict__.keys()]

    def __repr__(self) -> str:
        retstr = "Single frame metadata: \n"
        for key in self.__dict__:
            retstr += "\t" + str(key) + " : " + str(getattr(self,key)) + "\n"
        return retstr
