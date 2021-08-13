"""
A few tools for reading .siff and .tiff files from ScanImage that should be tucked away.

SCT March 28 2021, still rainy in Maywood
"""

from .registration import *
from .matlab_to_python import *
from .fluorophore_inits import available_fluorophores
from .slicefcns import *

# types to cast strings to when looked up
frame_meta_lookup_cast ={
    'FrameNumbers' : int,
    'frameNumberAcquisition' : int,
    'frameTimestamps_sec' : float,
    'epoch' : float
}

def most_important_header_data(header_dict):
    """ Returns a dict of the most important data.
    KEYS are strings, VALUES as defined below.
    TODO: Add all the values I want. I'm sure it'll
    come up as I do analysis

    RETURNS
    -------
    IM_PARAMS (dict):
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
    try:
        im_params["PICOSECONDS_PER_BIN"] = 10*2**(int(header_dict['SI.hScan2D.hAcq.binResolution']))
        im_params["NUM_BINS"] = int(header_dict['SI.hScan2D.hAcq.Tau_bins'])
    except:
        warnings.warn(
            """
            File lacks header data relating to PicoQuant MultiHarps -- this may be a non-FLIM
            ScanImage build. FLIM-dependent code may not work for these images!!
            """
        )

    return im_params

def get_color_ax(numpy_array):
    """
    Returns which axis is the color axis of input array (only works on standard order data)
    """
    if numpy_array.ndim < 4: return None
    if numpy_array.ndim == 4: return 0
    if numpy_array.ndim == 5: return 1
    if numpy_array.ndim == 6: return 2


### Dealing with framewise metadata

def line_to_dict_val(line):
    splitline = line.split(' = ')
    if splitline[0] in frame_meta_lookup_cast:
        return frame_meta_lookup_cast[splitline[0]](splitline[1])
    
def single_frame_metadata_to_dict(frame):
    return {line.split(' = ')[0]:line_to_dict_val(line) for line in frame['Frame metadata'].split('\n') if len(line.split(' = '))>0}

def frame_metadata_to_dict(metadata):
    """ Returns a list of metadata dicts from a list of siffreader metadata returns """
    return [single_frame_metadata_to_dict(frame) for frame in metadata]