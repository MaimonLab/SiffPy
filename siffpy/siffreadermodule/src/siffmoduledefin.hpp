#ifndef SIFFMODULEDEFIN_HPP
#define SIFFMODULEDEFIN_HPP


// KEYWORD ARGS BY FUNCTION

#define GET_FRAMES_KEYWORDS (char*[]){"frames", "type", "flim", NULL}

#define GET_FRAMES_METADATA_KEYWORDS (char*[]){"frames", NULL}

#define POOL_FRAMES_KEYWORDS (char*[]){"pool_lists", "type", "flim", NULL}

#define FLIM_MAP_KEYWORDS (char*[]){"params","frames", "confidence_metric", NULL}

#define GET_HISTOGRAM_KEYWORDS (char*[]){"frames", NULL}


// DOCSTRING DEFS

#define MODULE_DOC \
    "siffreader C extension module\n"\
    "Reads and interprets .siff and ScanImage .tiffs\n"\
    "\n"\
    "FUNCTIONS:\n"\
    "open(filename):\n"\
        "\tOpens the file at location filename.\n"\
    "close():\n"\
        "\tCloses an open file.\n"\
    "get_file_header():\n"\
        "\tReturns header data that applies across the file."\
    "get_frames(frames=[], type=list, flim=False):\n"\
        "\tReturns frames as a list of numpy arrays.\n"\
    "get_frame_metadata(frames=[]):\n"\
        "\tReturns frame metadata.\n"\
    "pool_frames(pool_lists, type=list, flim=False):\n"\
        "\tReturns summed versions of frames.\n"\
    "flim_map(params, framelist=None, confidence_metric='log_p'):\n"\
        "\tReturns a tuple: empirical lifetime, intensity, and a confidence metric.\n"\
    "get_histogram(frames=None):\n"\
        "\tReturns histogrm of photon arrival times."\
    "suppress_warnings():\n"\
        "\tSuppresses module-specific warnings.\n"\
    "report_warnings():\n"\
        "\tAllows module-specific warnings.\n"\
    "num_frames():\n"\
        "\tIf file is open, reports the total number of frames."


#define OPEN_DOCSTRING \
    "open(str filename)\n"\
    "Opens a .siff file, or even a .tiff file! Fancy that.\n"\
    "Input arguments:\n"\
    "\tfilename (str): path to a .tiff or .siff file."

#define CLOSE_DOCSTRING \
    "close()\n"\
    "Closes an open .siff or .tiff, if one is open.\n"\

#define GET_FILE_HEADER_DOCSTRING \
    "get_file_header()\n"\
    "Retrieves non-varying file data, e.g. ScanImage variable "\
    "values at the onset.\nReturns as a dict, with keywords corresponding "\
    "to ScanImage variables and values their corresponding measurements."

#define GET_FRAMES_DOCSTRING \
    "get_frames(frames=[], type=list, flim=False)\n"\
    "Reads an opened .siff or .tiff file and returns a PyObject corresponding to a (time by z by y by x by tau) array.\n"\
    "Input arguments:\n"\
    "\tframes (optional, LIST): a list of the indices of the frames to extract. "\
    "NOTE: Frames will be returned in the order they are listed! TODO: Make this timepoint-specific\n"\
    "\ttype (optional, TYPE): format of returned data. Can be list or numpy.ndarray. "\
    "If list, returns a list of single frame numpy arrays. If an ndarray, returns a full (time by color by z by x by y by tau) array, "\
    "or otherwise as specified. (NOT YET IMPLEMENTED)\n"\
    "\tflim (optional, BOOL): return a tau axis containing arrival time (irrelevant if file is not a .siff)"

#define GET_METADATA_DOCSTRING \
    "get_metadata(frames=[])\n"\
    "Get the framewise metadata, e.g. dimensions, timestamps.\n"\
    "Input arguments:\n"\
    "\tframes (optional, LIST): a list of the indices of the frames whose metadata is to be extract. "\
    "TODO: Make this timepoint-specific instead of frames?"

#define POOL_FRAMES_DOCSTRING \
    "pool_frames(pool_lists, type=list, flim=False)\n"\
    "Reads an opened .siff or .tiff file and returns a PyObject corresponding to frames pooled according to the pool_lists argument.\n"\
    "Each element of the returned list corresponds to one element of the pool_lists: the pooled version of the indices contained IN that element.\n"\
    "Input arguments:\n"\
    "\tpool_list (LIST (of lists, TODO make accept array-like)): a list of the indices of the frames to extract."\
    "e.g. [[4,5,6], [0,1,2], [0,8,4]]. Order of sublists does not matter for pooling."\
    "NOTE: Frames will be returned in the order they are listed! TODO: Make this timepoint-specific\n"\
    "\ttype (optional, TYPE): format of returned data. Can be list or numpy.ndarray."\
    "If list, returns a list of single frame numpy arrays. If an ndarray, returns a full (time by color by z by x by y by tau) array,"\
    "or otherwise as specified. (NOT YET IMPLEMENTED)\n"\
    "\tflim (optional, BOOL): return a tau axis containing arrival time (irrelevant if file is not a .siff)"

#define FLIM_MAP_DOCSTRING \
    "flim_map(params, framelist=None, confidence_metric='chi_sq')\n"\
    "Takes in a FLIMParams object and a list of lists of frames.\n"\
    "Returns a tuple of lists, each as long as the framelist.\n"\
    "Written to be (far) more memory efficient than using 3d numpy arrays.\n"\
    "Input arguments:\n"\
    "\tparams (FLIMParams): A fitted FLIMParams object giving the relevant info for the fluorophore.\n"\
    "\tframelist (list of lists of ints): Formatted as in pool_frames, the length of this list is"\
    "how many frames to return. Each sublist is the list of frames to pool together.\n"\
    "\tconfidence_metric (string): A string determining what the final element of the tuple will contain.\n"\
    "\tOptions:\n"\
    "\t\t'log_p'  : log likelihood of the pixel distribution (computes fast, iteratively).\n"\
    "\t\t'chi_sq' : chi-squared statistic of the data (computes slower, has some nicer properties).\n"\
    "Returns:\n"\
    "\t(flimmap, intensity, chi_sq)\n"\
    "\t\tflimmap: list of np.ndarrays containing the empirical lifetime for each pixel (UNITS OF HISTOGRAM BINS).\n"\
    "\t\tintensity: list of np.ndarrays containing the number of photons used to compute the lifetime in each pixel.\n"\
    "\t\tconfidence: list of np.ndarrays reporting the confidence metric for each pixel under the assumption in FLIMParams."

#define GET_HISTOGRAM_DOCSTRING \
    "get_histogram(frames=None)\n"\
    "Retrieves only the photon arrival times from the frames in the list frames.\n"\
    "Input arguments:\n"\
    "\tframes (optional, LIST): a list of the indices of the frames whose arrival times are desired.\n"\
    "\tif NONE, collects from ALL frames."\
    "Returns:\n"\
    "HISTOGRAM (ndarray): 1 dimensional numpy.ndarray"

#define SUPPRESS_WARNINGS_DOCSTRING \
    "suppress_warnings()\n"\
    "Suppresses output warnings for siffreader functions."

#define REPORT_WARNINGS_DOCSTRING \
    "report_warnings()\n"\
    "Forces reporting of warnings for siffreader functions."

#define NUM_FRAMES_DOCSTRING \
    "num_frames()\n"\
    "Reports number of frames in opened file."



#endif