#ifndef SIFFMODULEDEFIN_HPP
#define SIFFMODULEDEFIN_HPP


// KEYWORD ARGS BY FUNCTION
// NOTE: REGULAR ARGS HAVE TO BE HERE TOO!
#define GET_FRAMES_KEYWORDS (const char*[]){"frames", "type", "flim", "registration", "discard_bins", NULL}

#define GET_FRAMES_METADATA_KEYWORDS (const char*[]){"frames", NULL}

#define POOL_FRAMES_KEYWORDS (const char*[]){"pool_lists", "type", "flim", "registration", "discard_bins", NULL}

#define FLIM_MAP_KEYWORDS (const char*[]){"params","frames", "confidence_metric", "registration","sizeSafe", "discard_bins", NULL}

#define SUM_ROIS_KEYWORDS (const char*[]){"mask", "frames", "registration", NULL}

#define SUM_ROI_FLIM_KEYWORDS (const char*[]){"mask", "params", "frames", "registration", NULL}

#define GET_HISTOGRAM_KEYWORDS (const char*[]){"frames", NULL}

#define SIFF_TO_TIFF_KEYWORDS (const char*[]){"sourcepath", "savepath", NULL}


// DOCSTRING DEFS

#define MODULE_DOC \
    "siffreader C extension module\n"\
    "Reads and interprets .siff and ScanImage .tiffs\n"\
    "Can be used in one of two ways: either directly calling"\
    "functions from siffreader (siffreader.get_frames(*args, **kwargs)"\
    "or instantiating a siffreader.SiffIO object (preferred).\n"\
    "\n"\
    "FUNCTIONS:\n"\
    "open(filename):\n"\
        "\tOpens the file at location filename.\n"\
    "close():\n"\
        "\tCloses an open file.\n"\
    "get_file_header():\n"\
        "\tReturns header data that applies across the file.\n"\
    "get_frames(frames=[], type=list, flim=False, registration = None, discard_bins = None):\n"\
        "\tReturns frames as a list of numpy arrays.\n"\
    "get_frame_metadata(frames=[]):\n"\
        "\tReturns frame metadata.\n"\
    "pool_frames(pool_lists, type=list, flim=False, registration=None):\n"\
        "\tReturns summed versions of frames.\n"\
    "flim_map(params, framelist=None, confidence_metric='log_p', registration=None):\n"\
        "\tReturns a tuple: empirical lifetime, intensity, and a confidence metric.\n"\
    "sum_roi(mask : np.ndarray, frames : list[int] = None, registration : dict = None) -> np.ndarray:\n"\
        "\tReturns the sum of the photon counts within the provided ROI in the requested frames.\n"\
    "sum_roi_flim(mask : np.ndarray, params : siffpy.siffutils.flimparams.FLIMParams, "\
        "frames : list[int] = None, registration : dict = None) -> np.ndarray:\n"\
        "\tReturns the empirical lifetime estimated using the FLIMParams provided within the ROI in the requested frames.\n"\
    "get_histogram(frames=None):\n"\
        "\tReturns histogrm of photon arrival times."\
    "suppress_warnings():\n"\
        "\tSuppresses module-specific warnings.\n"\
    "report_warnings():\n"\
        "\tAllows module-specific warnings.\n"\
    "num_frames():\n"\
        "\tIf file is open, reports the total number of frames.\n"\
    "debug():\n"\
        "\tEnables siffreadermodule debugging log."\
    "sifftotiff(sourcepath : str, savepath : str = None):\n"\
        "\tConverts a .siff file (in sourcepath) to a .tiff file (saved in savepath), discarding arrival time information, if relevant."


#define OPEN_DOCSTRING \
    "open(filename : str)->None\n"\
    "Opens a .siff file, or even a .tiff file! Fancy that.\n"\
    "Input arguments:\n"\
    "\tfilename (str): \n\t\tPath to a .tiff or .siff file."

#define CLOSE_DOCSTRING \
    "close()->None\n"\
    "Closes an open .siff or .tiff, if one is open.\n"\

#define GET_FILE_HEADER_DOCSTRING \
    "get_file_header()->dict\n"\
    "Retrieves non-varying file data, e.g. ScanImage variable "\
    "values at the onset.\nReturns as a dict, with keywords corresponding "\
    "to ScanImage variables and values their corresponding measurements."

#define GET_FRAMES_DOCSTRING \
    "get_frames(frames : list[int] = [], type : type = list, flim : bool = False, registration : dict = {})\n"\
    "Reads an opened .siff or .tiff file and returns a PyObject corresponding to a (time by z by y by x by tau) array.\n"\
    "Input arguments:\n"\
    "\tframes (optional, LIST): \n\t\tA list of the indices of the frames to extract. "\
    "NOTE: Frames will be returned in the order they are listed! TODO: Make this timepoint-specific\n"\
    "\ttype (optional, TYPE): \n\t\tFormat of returned data. Can be list or numpy.ndarray. "\
    "If list, returns a list of single frame numpy arrays. If an ndarray, returns a full (time by color by z by x by y by tau) array, "\
    "or otherwise as specified. (NOT YET IMPLEMENTED)\n"\
    "\tflim (optional, BOOL): \n\t\treturn a tau axis containing arrival time (irrelevant if file is not a .siff)"\
    "\tregistration (optional, dict): a registration dictionary whose keys are the frame number (ints!) and whose values are rigid translations."\
    "\tdiscard_bins (optional, int): arrival bin (IN UNITS OF BIN) beyond which to discard photons"

#define GET_METADATA_DOCSTRING \
    "get_metadata(frames : list[int] = [])\n"\
    "Get the framewise metadata, e.g. dimensions, timestamps.\n"\
    "Input arguments:\n"\
    "\tframes (optional, LIST): \n \t\ta list of the indices of the frames whose metadata is to be extract. "\
    "TODO: Make this timepoint-specific instead of frames?"

#define POOL_FRAMES_DOCSTRING \
    "pool_frames(pool_lists : list[list[int]], type : type = list, flim : bool = False, registration : dict = None)\n"\
    "Reads an opened .siff or .tiff file and returns a PyObject corresponding to frames pooled according to the pool_lists argument.\n"\
    "Each element of the returned list corresponds to one element of the pool_lists: the pooled version of the indices contained IN that element.\n"\
    "Input arguments:\n"\
    "\tpool_list (LIST (of lists, TODO make accept array-like)): a list of the indices of the frames to extract."\
    "e.g. [[4,5,6], [0,1,2], [0,8,4]]. Order of sublists does not matter for pooling."\
    "NOTE: Frames will be returned in the order they are listed! TODO: Make this timepoint-specific\n"\
    "\ttype (optional, TYPE): format of returned data. Can be list or numpy.ndarray."\
    "If list, returns a list of single frame numpy arrays. If an ndarray, returns a full (time by color by z by x by y by tau) array,"\
    "or otherwise as specified. (NOT YET IMPLEMENTED)\n"\
    "\tflim (optional, BOOL): return a tau axis containing arrival time (irrelevant if file is not a .siff)\n"\
    "\tregistration (optional, dict): a registration dictionary whose keys are the frame number (ints!) and whose values are rigid translations."\
    "\tdiscard_bins (optional, int): arrival bin (IN UNITS OF BIN) beyond which to discard photons"

#define FLIM_MAP_DOCSTRING \
    "flim_map(params : siffutils.FLIMParams, framelist : list[list[int]] = None, confidence_metric : string = 'chi_sq')\n"\
    "Takes in a FLIMParams object and a list of lists of frames.\n"\
    "Returns a list of tuples, each with 3 elements.\n"\
    "Written to be (far) more memory efficient than using 3d numpy arrays.\n"\
    "Input arguments:\n"\
    "\tparams (FLIMParams): A fitted FLIMParams object giving the relevant info for the fluorophore.\n"\
    "\tframelist (list of lists of ints): Formatted as in pool_frames, the length of this list is"\
    "how many frames to return. Each sublist is the list of frames to pool together.\n"\
    "\tconfidence_metric (string): A string determining what the final element of the tuple will contain.\n"\
    "\tOptions:\n"\
    "\t\t'log_p'  : log likelihood of the pixel distribution (computes fast, iteratively).\n"\
    "\t\t'chi_sq' : chi-squared statistic of the data (computes slower, has some nicer properties).\n"\
    "\t\t'None'   : No confidence measure. Much faster.\n"\
    "\tregistration (optional, dict): a registration dictionary whose keys are the frame number (ints!) and whose values are rigid translations."\
    "Returns:\n"\
    "\t(flimmap, intensity, chi_sq)\n"\
    "\t\tflimmap: list of np.ndarrays containing the empirical lifetime for each pixel (UNITS OF HISTOGRAM BINS).\n"\
    "\t\tintensity: list of np.ndarrays containing the number of photons used to compute the lifetime in each pixel.\n"\
    "\t\tconfidence: list of np.ndarrays reporting the confidence metric for each pixel under the assumption in FLIMParams."\
    "\tdiscard_bins (optional, int): arrival bin (IN UNITS OF BIN) beyond which to discard photons"

#define SUM_ROI_DOCSTRING \
    "sum_roi(mask : np.ndarray, frames : list[int] = None, registration : dict = None) -> np.ndarray:\n"\
    "Requires a numpy array mask of dtype bool. Sums all the photons within the ROI for each of the frames requested"\
    "in the list frames. If frames is None, sums for all frames. Returns a 1d numpy array of dtype uint64 with length"\
    "equal to the number of frames requested.\n"\
    "Input arguments:\n"\
    "\tframes : list[int]\n"\
    "\t\tA list of the integer indices of the frames to sum within the ROI.\n"\
    "\tregistration : dict \n"\
    "\t\tA registration_dict object whose keys are ints and whose values are tuples corresponding"\
    "to a rigid shift in the y and x directions of the image."\
    "Returns:\n"\
    "\tsummed : np.ndarray\n"\
    "\t\t1d numpy array with length equal to the length of the number of frames requested."

#define SUM_ROI_FLIM_DOCSTRING \
    "sum_roi_flim(mask : np.ndarray, params : `siffpy.siffutils.flimparams.FLIMParams, "\
    "frames : list[int] = None, registration : dict = None) -> np.ndarray\n"\
    "Requires a numpy array mask of dtype bool and a FLIMParams object. Sums all the photons within the"\
    " provided ROI for each frame requested to compute an empirical lifetime for the ROI. If frames is None,"\
    " computes the sum for all frames. Returns a 1d numpy array of dtype float with length equal to the number of "\
    "frames requested. NOTE: the empirical lifetime is in units of TIME BINS.\n"\
    "Input arguments:\n"\
    "\tmask: np.ndarray of dtype bool\n"\
    "\tparams : `siffpy.siffutils.flimparams.FLIMParams`\n"\
    "\tframes : list[int]\n"\
    "\tregistration : dict\n"\
    "Returns:\n"\
    "\tsummed : np.ndarray\n"\
    "\t\t1d numpy array with length equal to the number of frames requested. Empirical lifetime measured in units of"\
    " time bins of the arrival time measuring device (e.g. MultiHarp)."

#define GET_HISTOGRAM_DOCSTRING \
    "get_histogram(frames : list[int] = None)-> np.ndarray \n"\
    "Retrieves only the photon arrival times from the frames in the list frames.\n"\
    "Input arguments:\n"\
    "\tframes (optional, LIST): \n\t\tA list of the indices of the frames whose arrival times are desired.\n"\
    "\tif NONE, collects from ALL frames."\
    "Returns:\n"\
    "HISTOGRAM (ndarray): 1 dimensional numpy.ndarray"

#define SUPPRESS_WARNINGS_DOCSTRING \
    "suppress_warnings()->None\n"\
    "Suppresses output warnings for siffreader functions."

#define REPORT_WARNINGS_DOCSTRING \
    "report_warnings()->None\n"\
    "Forces reporting of warnings for siffreader functions."

#define NUM_FRAMES_DOCSTRING \
    "num_frames()-> int\n"\
    "Reports number of frames in opened file."

#define DEBUG_DOCSTRING \
    "debug() -> None\n"\
    "Creates a debug log.\n"

#define SIFF_TO_TIFF_DOCSTRING \
    "siff_to_tiff(sourcepath : str, savepath : str = None) -> None\n"\
    "Converts a siff file located at sourcepath to a tiff file and saves it in location savepath"


#endif