#ifndef SIFFMODULEDEFIN_HPP
#define SIFFMODULEDEFIN_HPP


// KEYWORD ARGS BY FUNCTION

#define GET_FRAMES_KEYWORDS (char*[]){"frames", "type", "flim", NULL}

#define GET_FRAMES_METADATA_KEYWORDS (char*[]){"frames", NULL}

#define POOL_FRAMES_KEYWORDS (char*[]){"pool_lists", "type", "flim", NULL}


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
    "get_metadata(frames=[]):\n"\
        "\tReturns frame metadata.\n"\
    "pool_frames(pool_lists, type=list, flim=False):\n"\
        "\tReturns summed versions of frames.\n"\
    "suppress_warnings():\n"\
        "\tSuppresses module-specific warnings.\n"\
    "report_warnings():\n"\
        "\tAllows module-specific warnings."\
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