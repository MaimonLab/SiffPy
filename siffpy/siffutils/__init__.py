"""
A few tools for reading .siff and .tiff files from ScanImage that should be tucked away.

These are mostly just helper functions for image extraction specifically, not analysis
of the data itself.

Began
SCT March 28 2021, still rainy in Maywood

"""

from typing import Any
from .registration import *
from .matlab_to_python import *
from .fluorophore_inits import available_fluorophores
from .slicefcns import *
from .circle_fcns import *
from .imparams import ImParams

# types to cast strings to when looked up
frame_meta_lookup_cast ={
    'frameNumbers' : int,
    'frameNumberAcquisition' : int,
    'frameTimestamps_sec' : float,
    'epoch' : float,
    'endOfAcquisition' : bool,
    'endOfAcquisitionMode' : bool
}

MULTIHARP_BASE_RES = 5 # in picoseconds WARNING BEFORE MHLIB V3 THIS VALUE IS 2. I DIDN'T THINK TO PUT THIS INFO IN THE SIFF FILE

def most_important_header_data(header_dict):
    """ Returns a dict of the most important data.
    KEYS are strings, VALUES as defined below.
    TODO: Add all the values I want. I'm sure it'll
    come up as I do analysis

    RETURNS
    -------
    siffutils.imparams.ImParams:

        Attributes:
        ----------
        NUM_SLICES -- (int)
        FRAMES_PER_SLICE -- (int)
        STEP_SIZE -- (float)
        Z_VALS -- (list of floats)
        COLORS -- (list of ints)
        XSIZE -- (int)
        YSIZE -- (int)
        TODO:XRESOLUTION -- (float)
        TODO:YRESOLUTION -- (float)
        ZOOM -- (float)
        PICOSECONDS_PER_BIN -- (int)
        NUM_BINS -- (int)
    """

    im_params = {}
    im_params["NUM_SLICES"] = int(header_dict["SI.hStackManager.actualNumSlices"])
    im_params["FRAMES_PER_SLICE"] = int(header_dict["SI.hStackManager.framesPerSlice"])
    if im_params["NUM_SLICES"] > 1:
        im_params["STEP_SIZE"] = float(header_dict["SI.hStackManager.actualStackZStepSize"])
    im_params["Z_VALS"] = vector_to_list(header_dict['SI.hStackManager.zsRelative'], ret_type=float)
    im_params["COLORS"] = vector_to_list(header_dict["SI.hChannels.channelSave"], ret_type = int)
    im_params["ZOOM"] = float(header_dict['SI.hRoiManager.scanZoomFactor'])
    im_params["IMAGING_FOV"] = matrix_to_listlist(header_dict['SI.hRoiManager.imagingFovUm'])
    try:
        im_params["PICOSECONDS_PER_BIN"] = MULTIHARP_BASE_RES*2**(int(header_dict['SI.hScan2D.hAcq.binResolution']))
        im_params["NUM_BINS"] = int(header_dict['SI.hScan2D.hAcq.Tau_bins'])
    except:
        warnings.warn(
            """
            File lacks header data relating to PicoQuant MultiHarps -- this may be a non-FLIM
            ScanImage build. FLIM-dependent code may not work for these images!!
            """
        )

    return ImParams(im_params)

def get_color_ax(numpy_array):
    """
    Returns which axis is the color axis of input array (only works on standard order data)
    """
    if numpy_array.ndim < 4: return None
    if numpy_array.ndim == 4: return 0
    if numpy_array.ndim == 5: return 1
    if numpy_array.ndim == 6: return 2


### Dealing with framewise metadata

def line_to_dict_val(line : str) -> str:
    splitline = line.split(' = ')
    if splitline[0] in frame_meta_lookup_cast:
        return frame_meta_lookup_cast[splitline[0]](splitline[1])
    elif len(splitline) == 2:
        return splitline[1]
    elif len(splitline) > 2:
        return ' = '.join(splitline[1:]) # reconcatenate the content of these ones
    else:
        return line
    
def single_frame_metadata_to_dict(frame : dict) -> dict:
    """
    Each list item returned by siffreader.get_frame_metadata
    is a dict, one entry in which corresponds to the string
    of metadata that is written to each IFD about that frame
    (e.g. timestamps). This plucks out that string, parses each
    line, and returns it a dict with just those pieces of metadata
    """
    return {line.split(' = ')[0]:line_to_dict_val(line) for line in frame['Frame metadata'].split('\n') if len(line.split(' = '))>0}

def frame_metadata_to_dict(metadata : list[dict]) -> list[dict]:
    """ Returns a list of metadata dicts from a list of siffreader metadata returns """
    return [single_frame_metadata_to_dict(frame) for frame in metadata]

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
