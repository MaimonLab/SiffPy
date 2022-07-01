import logging, re

from ..utils import ImParams

MULTIHARP_BASE_RES = 5 # in picoseconds WARNING BEFORE MHLIB V3 THIS VALUE IS 20. I DIDN'T THINK TO PUT THIS INFO IN THE SIFF FILE

def vector_to_list(vector, vec_num : int = 0, ret_type=float):
    """
    list = vector_to_list(vector, type=float)

    Interprets either a MATLAB column vector or row vector as a python list

    Inputs
    ------
    vector (string):
        String version of a MATLAB vector, e.g. '[1;2]' or '[1,5,5]'

    vec_num (int, optional):
        Which vector element to use.

    type (optional):
        type of numbers (int, float)

    Returns
    ------
    list (list):
        List version of the vector input
    """
    # if it's just a number, then we don't need to worry about this
    try:
        return ret_type(vector)
    except:
        pass
    
    betwixt_brackets = re.findall(r"^.*\[(.*)\].*$",vector)
    if not betwixt_brackets:
        return None
    if len(betwixt_brackets) > 1 and vec_num == 0:
        logging.warning("Ambiguous string. Using first matching vector.")
    col_split = betwixt_brackets[vec_num].split(';')
    row_split = betwixt_brackets[vec_num].split(' ')

    if (len(col_split) > 1) and (len(row_split) > 1):
        raise ValueError("Input string could not be parsed into a row vector or column vector.")
    if len(col_split)>1:
        return [ret_type(element) for element in col_split]
    else:
        return [ret_type(element) for element in row_split]

def matrix_to_listlist(matrix : str, vec_num : int = 0, ret_type = float) -> list[list]:
    """
    Converts the string representation of a MATLAB matrix into a list of lists
    """
    try:
        return ret_type(matrix)
    except:
        pass
    
    betwixt_brackets = re.findall(r"^.*\[(.*)\].*$",matrix)
    if not betwixt_brackets:
        return None
    if len(betwixt_brackets) > 1 and vec_num == 0:
        logging.warning("Ambiguous string. Using first matching vector.")
    col_split = betwixt_brackets[vec_num].split(';')
    row_split = betwixt_brackets[vec_num].split(' ')

    if (len(col_split) > 1) and (len(row_split) > 1):
        return [[ret_type(element) for element in column.split(" ")] for column in col_split]
    # if it's just a vector, use the vector parser
    else:
        return vector_to_list(matrix, ret_type)
    
def header_data_to_nvfd(hd):
    return {entry.split(" = ")[0] : (entry.split(" = ")[1] if (len(entry.split(" = "))>1) else None) for entry in hd["Non-varying frame data"].split("\n")}

def header_data_to_roi_string(hd : str) -> dict:
    """ Iterate through the many layers of the ROI strings to return the appropriate dict """
    return eval(hd['ROI string'].replace("null", "None"))

def header_to_imparams(header : str, num_frames : int = None)->ImParams:
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
    header_dict = header_data_to_nvfd(header)
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
        logging.warning(
            """
            File lacks header data relating to PicoQuant MultiHarps -- this may be a non-FLIM
            ScanImage build. FLIM-dependent code may not work for these images!!
            """
        )

    ROI_group_data = header_data_to_roi_string(header)
    try:
        xy = ROI_group_data['RoiGroups']['imagingRoiGroup']['rois']['scanfields']['pixelResolutionXY']
        im_params["XSIZE"] = xy[0]
        im_params["YSIZE"]= xy[1]
    except:
        raise Exception("ROI header information is more complicated. Probably haven't implemented the reader"
        " to be comaptible with mROI scanning. Don't worry -- if you're getting this error, I'm already"
        " planning on addressing it."
        )
    finally:
        im_params['NUM_FRAMES'] = num_frames
        im_pars = ImParams(**im_params)
        return im_pars