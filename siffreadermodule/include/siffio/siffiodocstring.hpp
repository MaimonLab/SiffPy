#ifndef SIFFIO_DOCSTRING_HPP
#define SIFFIO_DOCSTRING_HPP

#define SIFFIO_OBJECTNAME "SiffIO"
#define SIFFIO_TPNAME "siffreader.SiffIO"
#define SIFFIO_DOCSTRING "SiffIO -- Work in progress"

PyDoc_STRVAR(
    siffio_open_doc,
    "SiffIO.open(filename : str)->None\n"
    "--\n"
    "\n"
    "Opens a file in C++ to be read and return "
    "numpy arrays.\n\n"
    "Arguments:\n\n"
    "filename : str\n"
    "\tAbsolute path to the file to open."
);

PyDoc_STRVAR(
    siffio_close_doc,
    "SiffIO.close()->None\n"
    "--\n"
    "\n"
    "Closes the open file (if there is one)."
);

PyDoc_STRVAR(
    siffio_get_file_header_doc,
    "SiffIO.get_file_header()->dict\n"
    "--\n"
    "\n"
    "Retrieves non-varying file data, e.g. ScanImage variable "
    "values at the onset.\nReturns as a dict, with keywords corresponding "
    "to ScanImage variables and values their corresponding measurements."
);

PyDoc_STRVAR(
    siffio_num_frames_doc,
    "SiffIO.num_frames()-> int\n"
    "--\n"
    "\n"
    "Reports number of frames in opened file."
);

PyDoc_STRVAR(
    siffio_get_experiment_timestamps_doc,
    "SiffIO.get_experiment_timestamps()->np.ndarray[Any, np._float]\n"
    "--\n"
    "\n"
    "Retrieves the timestamps of each frame computed using"
    " the number of laser pulses since the start of the experiment."
    "\n"
    "Arguments:\n\n"
    "\tframes : list[int] \n\t\tA list of the indices of the frames whose timestamps are desired. "
    "\n"
    "Returns:\n\n"
    "\ttimestamps : np.ndarray[Any, np._float]\n\t\t"
    "A numpy array of timestamps in _experiment_ time (seconds since start)."
);

PyDoc_STRVAR(
    siffio_get_epoch_timestamps_laser_doc,
    "SiffIO.get_epoch_timestamps_laser()->np.ndarray[Any, np._uint64]\n"
    "--\n"
    "\n"
    "Retrieves the timestamps of each frame computed using"
    " the number of laser pulses since the start of the experiment."
    "\n"
    "Arguments:\n\n"
    "\tframes : list[int] \n\t\tA list of the indices of the frames whose timestamps are desired. "
    "\n"
    "Returns:\n\n"
    "\ttimestamps : np.ndarray[Any, np._uint64]\n\t\t"
    "A numpy array of timestamps in _epoch_ time (UTC)."
);

PyDoc_STRVAR(
    siffio_get_epoch_timestamps_system_doc,
    "SiffIO.get_epoch_timestamps_system()->np.ndarray[Any, np._uint64]\n"
    "--\n"
    "\n"
    "Retrieves the timestamps of each frame computed using"
    " the system clock."
    "\n"
    "Arguments:\n\n"
    "\tframes : list[int] \n\t\tA list of the indices of the frames whose timestamps are desired. "
    "\n"
    "Returns:\n\n"
    "\ttimestamps : np.ndarray[Any, np._uint64]\n\t\t"
    "A numpy array of timestamps in _epoch_ time (UTC)."
);

PyDoc_STRVAR(
    siffio_get_epoch_timestamps_both_doc,
    "SiffIO.get_epoch_both)->np.ndarray[Any, np._uint64]\n"
    "--\n"
    "\n"
    "Retrieves the timestamps of each frame computed using"
    " the system clock."
    "\n"
    "Arguments:\n\n"
    "\tframes : list[int] \n\t\tA list of the indices of the frames whose timestamps are desired. "
    "\n"
    "Returns:\n\n"
    "\ttimestamps : np.ndarray[Any, np._uint64]\n\t\t"
    "A numpy array of timestamps in _epoch_ time (UTC), with the first row being the laser clock"
    " and the second row being the system clock."
);

PyDoc_STRVAR(
    siffio_get_frames_doc,
    "SiffIO.get_frames(frames : list[int] = [], flim : bool = False, registration : dict = {})->list[np.ndarray]\n"
    "--\n"
    "\n"
    "Reads an opened .siff or .tiff file and returns a PyObject corresponding to a (time by z by y by x by tau) array.\n"
    "\n"
    "Arguments:\n\n"
    "\tframes : list[int] \n\t\tA list of the indices of the frames to extract. "
    "NOTE: Frames will be returned in the order they are listed! TODO: Make this timepoint-specific\n"
    "\trettype : type \n\t\tFormat of returned data. Can be list or numpy.ndarray. "
    "If list, returns a list of single frame numpy arrays. If an ndarray, returns a full (time by color by z by x by y by tau) array, "
    "or otherwise as specified. (NOT YET IMPLEMENTED)\n"
    "\tflim : bool = False \n\t\treturn a tau axis containing arrival time (irrelevant if file is not a .siff)"
    "\tregistration : dict = None \n\t\tA registration dictionary whose keys are the frame number (ints!) and whose values are rigid translations."
);

PyDoc_STRVAR(
    siffio_get_frame_metadata_doc,
    "SiffIO.get_metadata(frames : list[int] = [])->dict\n"
    "--\n"
    "\n"
    "Get the framewise metadata, e.g. dimensions, timestamps.\n"
    "\n"
    "Arguments:\n\n"
    "\tframes : list[int] \n \t\ta list of the indices of the frames whose metadata is to be extract. "
    "TODO: Make this timepoint-specific instead of frames?"
);

PyDoc_STRVAR(
    siffio_pool_frames_doc,
    "SiffIO.pool_frames(pool_lists : list[list[int]], type : type = list, flim : bool = False, registration : dict = None)->list[np.ndarray]\n"\
    "--\n"
    "\n"
    "Reads an opened .siff or .tiff file and returns a PyObject corresponding to frames pooled according to the pool_lists argument.\n"
    "Each element of the returned list corresponds to one element of the pool_lists: the pooled version of the indices contained IN that element.\n"
    "Arguments:\n\n"
    "\tpool_list (LIST (of lists, TODO make accept array-like)): a list of the indices of the frames to extract."
    "e.g. [[4,5,6], [0,1,2], [0,8,4]]. Order of sublists does not matter for pooling."
    "NOTE: Frames will be returned in the order they are listed! TODO: Make this timepoint-specific\n"
    "\ttype (optional, TYPE): format of returned data. Can be list or numpy.ndarray."
    "If list, returns a list of single frame numpy arrays. If an ndarray, returns a full (time by color by z by x by y by tau) array,"
    "or otherwise as specified. (NOT YET IMPLEMENTED)\n"
    "\tflim (optional, BOOL): return a tau axis containing arrival time (irrelevant if file is not a .siff)\n"
    "\tregistration (optional, dict): a registration dictionary whose keys are the frame number (ints!) and whose values are rigid translations."
    "\tdiscard_bins (optional, int): arrival bin (IN UNITS OF BIN) beyond which to discard photons"
);

PyDoc_STRVAR(
    siffio_flim_map_doc,
    "SiffIO.flim_map(params : FLIMParams, framelist : list[list[int]] = None, confidence_metric : string = 'chi_sq')->list[tuple[np.ndarray]]\n"
    "--\n"
    "\n"
    "Takes in a FLIMParams object and a list of lists of frames.\n"
    "Returns a list of tuples, each with 3 elements.\n"
    "Written to be (far) more memory efficient than using 3d numpy arrays.\n"
    "Input arguments:\n"
    "\tparams (FLIMParams): A fitted FLIMParams object giving the relevant info for the fluorophore.\n"
    "\tframelist (list of lists of ints): Formatted as in pool_frames, the length of this list is"
    "how many frames to return. Each sublist is the list of frames to pool together.\n"
    "\tconfidence_metric (string): A string determining what the final element of the tuple will contain.\n"
    "\tOptions:\n"
    "\t\t'log_p'  : log likelihood of the pixel distribution (computes fast, iteratively).\n"
    "\t\t'chi_sq' : chi-squared statistic of the data (computes slower, has some nicer properties).\n"
    "\t\t'None'   : No confidence measure. Much faster.\n"
    "\tregistration (optional, dict): a registration dictionary whose keys are the frame number (ints!) and whose values are rigid translations."
    "Returns:\n"
    "\t(flimmap, intensity, chi_sq)\n"
    "\t\tflimmap: list of np.ndarrays containing the empirical lifetime for each pixel (UNITS OF HISTOGRAM BINS).\n"
    "\t\tintensity: list of np.ndarrays containing the number of photons used to compute the lifetime in each pixel.\n"
    "\t\tconfidence: list of np.ndarrays reporting the confidence metric for each pixel under the assumption in FLIMParams."
    "\tdiscard_bins (optional, int): arrival bin (IN UNITS OF BIN) beyond which to discard photons"
);

PyDoc_STRVAR(
    siffio_sum_roi_doc,
    "SiffIO.sum_roi(mask : np.ndarray, frames : list[int] = None, registration : dict = None) -> np.ndarray:\n"
    "--\n"
    "\n"
    "Requires a numpy array mask of dtype bool. Sums all the photons within the ROI for each of the frames requested"
    "in the list frames. If frames is None, sums for all frames. Returns a 1d numpy array of dtype uint64 with length"
    "equal to the number of frames requested.\n"
    "Input arguments:\n"
    "\tframes : list[int]\n"
    "\t\tA list of the integer indices of the frames to sum within the ROI.\n"
    "\tregistration : dict \n"
    "\t\tA registration_dict object whose keys are ints and whose values are tuples corresponding"
    "to a rigid shift in the y and x directions of the image."
    "Returns:\n"
    "\tsummed : np.ndarray\n"
    "\t\t1d numpy array with length equal to the length of the number of frames requested."
);

PyDoc_STRVAR(
    siffio_sum_rois_doc,
    "SiffIO.sum_rois(masks : Union[List[np.ndarray],np.ndarray], frames : list[int] = None, registration : dict = None) -> np.ndarray\n"
    "--\n"
    "\n"
    "Requires a numpy array mask of dtype bool. Sums all the photons within the ROI for each of the frames requested"
    "in the list frames. If frames is None, sums for all frames. Returns a 2d numpy array of dtype uint64 with dimensions"
    "`(len(frames), masks.shape[0])`.\n"
    "Input arguments:\n"
    "\tmasks : Union[List[np.ndarray],np.ndarray]\n"
    "\t\tA list of numpy arrays of dtype bool, each corresponding to a different ROI.\n"
    "\tframes : list[int]\n"
    "\t\tA list of the integer indices of the frames to sum within the ROI.\n"
    "\tregistration : dict \n"
    "\t\tA registration_dict object whose keys are ints and whose values are tuples corresponding"
    "to a rigid shift in the y and x directions of the image."
    "Returns:\n"
    "\tsummed : np.ndarray\n"
    "\t\t2d numpy array with dimensions `(masks.shape[0], len(frames))`."
);

PyDoc_STRVAR(
    siffio_sum_roi_flim_doc,
    "SiffIO.sum_roi_flim(mask : np.ndarray, params : `siffpy.core.flim.flimparams.FLIMParams, "
    "frames : list[int] = None, registration : dict = None) -> np.ndarray\n"
    "--\n"
    "\n"
    "Requires a numpy array mask of dtype bool and a FLIMParams object. Sums all the photons within the"
    " provided ROI for each frame requested to compute an empirical lifetime for the ROI. If frames is None,"
    " computes the sum for all frames. Returns a 1d numpy array of dtype float with length equal to the number of "
    "frames requested. NOTE: the empirical lifetime is in units of TIME BINS.\n"
    "Arguments:\n\n"
    "\tmask: np.ndarray of dtype bool\n"
    "\tparams : `siffpy.core.flim.flimparams.FLIMParams`\n"
    "\tframes : list[int]\n"
    "\tregistration : dict\n"
    "Returns:\n"
    "\tsummed : np.ndarray\n"
    "\t\t1d numpy array with length equal to the number of frames requested. Empirical lifetime measured in units of"
    " time bins of the arrival time measuring device (e.g. MultiHarp)."
);

PyDoc_STRVAR(
    siffio_sum_rois_flim_doc,
    "SiffIO.sum_rois_flim(masks : Union[List[np.ndarray],np.ndarray], params : `siffpy.core.flim.flimparams.FLIMParams, "
    "frames : list[int] = None, registration : dict = None) -> np.ndarray\n"
    "--\n"
    "\n"
    "Requires a numpy array mask of dtype bool and a FLIMParams object. Sums all the photons within the"
    " provided ROI for each frame requested to compute an empirical lifetime for the ROI. If frames is None,"
    " computes the sum for all frames. Returns a 2d numpy array of dtype float with dimensions"
    "`(len(frames), masks.shape[0])`. NOTE: the empirical lifetime is in units of TIME BINS.\n"
    "Arguments:\n\n"
    "\tmasks: Union[List[np.ndarray],np.ndarray]\n"
    "\tparams : `siffpy.core.flim.flimparams.FLIMParams`\n"
    "\tframes : list[int]\n"
    "\tregistration : dict\n"
    "Returns:\n"
    "\tsummed : np.ndarray\n"
    "\t\t2d numpy array with dimensions `(masks.shape[0], len(frames))`. Empirical lifetime measured in units of"
    " time bins of the arrival time measuring device (e.g. MultiHarp)."
);

PyDoc_STRVAR(
    siffio_get_histogram_doc,
    "SiffIO.get_histogram(frames : list[int] = None)-> np.ndarray \n"
    "--\n"
    "\n"
    "Retrieves only the photon arrival times from the frames in the list frames.\n"
    "Arguments:\n\n"
    "\tframes (optional, LIST): \n\t\tA list of the indices of the frames whose arrival times are desired.\n"
    "\tif NONE, collects from ALL frames."
    "Returns:\n"
    "HISTOGRAM (ndarray): 1 dimensional numpy.ndarray"
);

PyDoc_STRVAR(
    siffio_get_appended_text_doc,
    "SiffIO.get_appended_text(frames : Optional[List[int]] = None)->List[str]\n"
);

#endif