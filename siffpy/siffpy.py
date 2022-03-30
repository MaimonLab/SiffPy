from ast import Not
import itertools
import logging, os, pickle
from .siffutils import SEC_TO_NANO, NANO_TO_SEC
from .siffutils.slicefcns import framelist_by_color

import numpy as np

import siffreader
from . import siffutils
from .siffutils.exp_math import *
from .siffutils.flimparams import FLIMParams
from .siffutils.flimarray import FlimArray
from .siffutils.typecheck import *
from .siffutils import registration
from .siffutils.events import parseMetaAsEvents

# TODO:
# __repr__
# ret_type in multiple functions
# "map to standard order"

__all__ = ['SiffReader','fit_exp']

class SiffReader(object):
    """
	Centralized Pythonic interface to the SiffReader module implemented in C.

    Designed to streamline several types of operations one might perform with a
    single file, so it operates by opening a file and then taking arguments that
    relate to that specific file, e.g. frame numbers or slice numbers, as opposed
    to accepting numpy arrays of frames.

	DOCSTRING still in progress. File first made SCT 03/24/2021

	...
	Attributes
	----------
    file_header (dict):
        Dictionary whose keys correspond to useful pan-file metadata, e.g. microscope settings.
    
    im_params (dict):
        The most useful meta data. Color channels, number of slices, number of volumes, z positions, etc.

    flim (bool):
        Whether or not to use arrival time data

    opened (bool):
        Is there an open .siff or .tiff?

    filename (str):
        If there is an open .siff or .tiff, this is its path

	Methods
	-------
	open(self, filename):
		Opens a .siff or .tiff file with path "filename".

    params = fit_exp(numpy_array, num_components=2)
        Takes a numpy array with dimensions (time, color, z, y,x,tau) or excluding any dimensions up to (y,x,tau) and
        returns a color-element list of dicts with fits of the fluorescence emission model for each color channel

    reshaped = map_to_standard_order(numpy_array, map_list=['time','color','z','y','x','tau'])
        Takes the numpy array numpy_array and returns it in the order (time, color, z, y, x, tau)

    """

### INIT AND DUNDER METHODS
    def __init__(self, filename : str = None):
        self.file_header = {}
        self.im_params = {}
        self.ROI_group_data = {}
        self.opened = False
        self.registration_dict = None
        self.reference_frames = None
        self.debug = False
        self.events = None
        if filename is None:
            self.filename = ''
        else:
            self.open(filename)

    def __del__(self):
        """ Close file before being destroyed """
        self.close()

    def __str__(self):
        ret_string = "SIFFREADER object:\n\n"
        if self.opened:
            ret_string += "Open file: "
            ret_string += self.filename
            ret_string += "\n"
        else:
            ret_string += "Inactive siffreader\n"
        if hasattr(self, "im_params"):
            ret_string += self.im_params.__repr__()
        if hasattr(self, "registrationDict"):
            ret_string += "Registration dict loaded\n"
        if hasattr(self, "reference_frames"):
            ret_string += "Reference images loaded\n"
        return ret_string
    
    def __repr__(self):
        # TODO
        return self.__str__()

    def open(self, filename: str = None) -> None:
        """
        Opens a .siff or .tiff file with path "filename". If no value provided for filename, prompts with a file dialog.
        INPUTS
        ------
        filename (optional):
            (str) path to a .siff or .tiff file.

        RETURN
        ------
        NONE
        """
        if filename is None:
            raise NotImplementedError("Dialog to navigate to file not yet implemented")
            
        if self.opened and not (filename == self.filename):
            siffreader.close()

        siffreader.open(filename)
        self.filename = filename

        hd = siffreader.get_file_header()
        self.file_header =  siffutils.header_data_to_nvfd(hd)
        
        if self.debug:
            print("Header read")
        
        self.im_params = siffutils.most_important_header_data(self.file_header)
        self.im_params['NUM_FRAMES'] = siffreader.num_frames()

        self.ROI_group_data = siffutils.header_data_to_roi_string(hd)
        self.opened = True

        try:
            xy = self.ROI_group_data['RoiGroups']['imagingRoiGroup']['rois']['scanfields']['pixelResolutionXY']
            self.im_params['XSIZE'] = xy[0]
            self.im_params['YSIZE'] = xy[1]
        except:
            raise Exception("ROI header information is more complicated. Probably haven't implemented the reader"
            " to be comaptible with mROI scanning. Don't worry -- if you're getting this error, I'm already"
            " planning on addressing it."
            )

        if os.path.exists(os.path.splitext(filename)[0] + ".dict"):
            with open(os.path.splitext(filename)[0] + ".dict", 'rb') as dict_file:
                reg_dict = pickle.load(dict_file)
            if isinstance(reg_dict, dict):
                logging.warning("\n\n\tFound a registration dictionary for this image and importing it.\n")
                self.registration_dict = reg_dict
            else:
                logging.warning("\n\n\tPutative registration dict for this file is not of type dict.\n")
        if os.path.exists(os.path.splitext(filename)[0] + ".ref"):
            with open(os.path.splitext(filename)[0] + ".ref", 'rb') as images_list:
                ref_ims = pickle.load(images_list)
            if isinstance(ref_ims, list):
                logging.warning("\n\n\tFound a reference image list for this file and importing it.\n")
                self.reference_frames = ref_ims
            else:
                logging.warning("\n\n\tPutative reference images object for this file is not of type list.\n", stacklevel=2)
        
        events = self.find_events()
        if len(events):
            self.events = events

    def close(self) -> None:
        """ Closes opened file """
        siffreader.close()
        self.opened = False
        self.filename = ''

### LOADING SAVED RELEVANT DATA METHODS
    def assign_registration_dict(self, path : str = None):
        """
        Assign a .dict file, overrides the automatically opened one.

        If no path is provided, looks for one with the same name as the opened filename
        """
        if path is None:
            if os.path.exists(os.path.splitext(self.filename)[0] + ".dict"):
                path = os.path.splitext(self.filename)[0] + ".dict"

        if not os.path.splitext(path)[-1] == '.dict':
            raise TypeError("File must be of extension .dict")
        
        with open(path, 'rb') as dict_file:
            reg_dict = pickle.load(dict_file)
        if isinstance(reg_dict, dict):
            self.registration_dict = reg_dict
        else:
            logging.warning("\n\n\tPutative registration dict for this file is not of type dict.\n")

    def load_reference_frames(self, path : str = None):
        """
        Assign a .ref file, overrides the automatically opened one.

        If no path is provided, looks for one with the same name as the opened filename
        """
        if path is None:
            if os.path.exists(os.path.splitext(self.filename)[0] + ".ref"):
                path = os.path.splitext(self.filename)[0] + ".ref"

        if not os.path.splitext(path)[-1] == '.ref':
            raise TypeError("File must be of extension .ref")
        
        with open(path, 'rb') as images_list:
            ref_ims = pickle.load(images_list)
        if isinstance(ref_ims, list):
            self.reference_frames = ref_ims
        else:
            logging.warning("\n\nPutative reference images object for this file is not of type list.\n", stacklevel=2)

### TIME AXIS METHODS
    def t_axis(self, timepoint_start : int = 0, timepoint_end : int = None, reference_z : int = 0) -> np.ndarray:
        """
        Returns the time-stamps of frames. By default, returns the time stamps of all frames relative
        to acquisition start.

        INPUTS
        ------
        
        timepoint_start (optional, int):
            Index of time point to start at. Note this is in units of
            TIMEPOINTS so this goes by step sizes of num_slices * num_colors!
            If no argument is given, this is treated as 0
        
        timepoint_end (optional, int):
            Index of time point to end at. Note Note this is in units of
            TIMEPOINTS so this goes by step sizes of num_slices * num_colors!
            If no argument is given, this is the end of the file.

        reference_z (optional, int):
            Picks the timepoint of a single slice in a z stack and only returns
            that corresponding value. Means nothing if imaging is a single plane.
            If no argument is given, assumes the first slice

        RETURN VALUES
        -------
        timepoints (1-d ndarray):
            Time point of the requested frames, relative to beginning of the
            image acquisition. Size will be:
                (timepoint_end - timepoint_start)*num_slices
            unless reference_z is used.
        """
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")
        num_slices = self.im_params['NUM_SLICES']
        
        num_colors = 1
        if self.im_params is list:
            num_colors = len(self.im_params['COLORS'])

        fps = self.im_params['FRAMES_PER_SLICE']
        
        timestep_size = num_slices*num_colors*fps # how many frames constitute a timepoint
        
        # figure out the start and stop points we're analyzing.
        frame_start = timepoint_start * timestep_size
        
        if timepoint_end is None:
            frame_end = self.im_params['NUM_FRAMES']
        else:
            if timepoint_end > self.im_params['NUM_FRAMES']//timestep_size:
                logging.warning(
                    "\ntimepoint_end greater than total number of frames.\n"+
                    "Using maximum number of complete timepoints in image instead.\n"
                )
                timepoint_end = self.im_params['NUM_FRAMES']//timestep_size
            
            frame_end = timepoint_end * timestep_size

        # now convert to a list of all the frames whose metadata we want
        framelist = [frame for frame in range(frame_start, frame_end) 
            if (((frame-frame_start) % timestep_size) == (num_colors*reference_z))
        ]
        
        return np.array([frame['frameTimestamps_sec']
            for frame in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=framelist))
        ])
    
    def get_time(self, frames : list[int] = None, reference : str = "experiment") -> np.ndarray:
        """
        Gets the recorded time (in seconds) of the frame(s) numbered in list frames

        INPUTS
        ------
        frames (optional, list):
            If not provided, retrieves time value of ALL frames.

        reference (optional, str):
            Referenced to start of the experiment, or referenced to epoch time.

            Possible values:
                experiment - referenced to experiment
                epoch      - referenced to epoch

        RETURN VALUES
        -------------
        time (np.ndarray):
            Ordered like the list in frames (or in order from 0 to end if frames is None).
            Time into acquisition of frames (in seconds)
        """
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")
        if frames is None:
            frames = list(range(self.im_params['NUM_FRAMES']))

        reference = reference.lower() # case insensitive

        if reference == "epoch":
            return np.array([frame['epoch'] # WARNING, OLD VERSIONS USED SECONDS NOT NANOSECONDS 
                for frame in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=frames))
            ])
        
        if reference == "experiment":
            return np.array([frame['frameTimestamps_sec']
                for frame in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=frames))
            ])
        else:
            ValueError("Reference argument not a valid parameter (must be 'epoch' or 'experiment')")

### METADATA METHODS
    def get_frames_metadata(self, frames : list[int] = None) -> list[siffutils.FrameMetaData]:
        if frames is None:
            frames = list(range(self.im_params['NUM_FRAMES']))
        return [siffutils.FrameMetaData(meta_dict)
            for meta_dict in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=frames))
        ]

    def t_axis(self, timepoint_start : int = 0, timepoint_end : int = None, reference_z : int = 0) -> np.ndarray:
        """
        Returns the time-stamps of frames. By default, returns the time stamps of all frames relative
        to acquisition start.

        INPUTS
        ------
        
        timepoint_start (optional, int):
            Index of time point to start at. Note this is in units of
            TIMEPOINTS so this goes by step sizes of num_slices * num_colors!
            If no argument is given, this is treated as 0
        
        timepoint_end (optional, int):
            Index of time point to end at. Note Note this is in units of
            TIMEPOINTS so this goes by step sizes of num_slices * num_colors!
            If no argument is given, this is the end of the file.

        reference_z (optional, int):
            Picks the timepoint of a single slice in a z stack and only returns
            that corresponding value. Means nothing if imaging is a single plane.
            If no argument is given, assumes the first slice

        RETURN VALUES
        -------
        timepoints (1-d ndarray):
            Time point of the requested frames, relative to beginning of the
            image acquisition. Size will be:
                (timepoint_end - timepoint_start)*num_slices
            unless reference_z is used.
        """
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")
        num_slices = self.im_params['NUM_SLICES']
        
        num_colors = 1
        if self.im_params is list:
            num_colors = len(self.im_params['COLORS'])

        fps = self.im_params['FRAMES_PER_SLICE']
        
        timestep_size = num_slices*num_colors*fps # how many frames constitute a timepoint
        
        # figure out the start and stop points we're analyzing.
        frame_start = timepoint_start * timestep_size
        
        if timepoint_end is None:
            frame_end = self.im_params['NUM_FRAMES']
        else:
            if timepoint_end > self.im_params['NUM_FRAMES']//timestep_size:
                logging.warning(
                    "\ntimepoint_end greater than total number of frames.\n"+
                    "Using maximum number of complete timepoints in image instead.\n"
                )
                timepoint_end = self.im_params['NUM_FRAMES']//timestep_size
            
            frame_end = timepoint_end * timestep_size

        # now convert to a list of all the frames whose metadata we want
        framelist = [frame for frame in range(frame_start, frame_end) 
            if (((frame-frame_start) % timestep_size) == (num_colors*reference_z))
        ]
        
        return np.array([frame['frameTimestamps_sec']
            for frame in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=framelist))
        ])
    
    def get_time(self, frames : list[int] = None, reference : str = "experiment") -> np.ndarray:
        """
        Gets the recorded time (in seconds) of the frame(s) numbered in list frames

        INPUTS
        ------
        frames (optional, list):
            If not provided, retrieves time value of ALL frames.

        reference (optional, str):
            Referenced to start of the experiment, or referenced to epoch time.

            Possible values:
                experiment - referenced to experiment
                epoch      - referenced to epoch

        RETURN VALUES
        -------------
        time (np.ndarray):
            Ordered like the list in frames (or in order from 0 to end if frames is None).
            Time into acquisition of frames (in seconds)
        """
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")
        if frames is None:
            frames = list(range(self.im_params['NUM_FRAMES']))

        reference = reference.lower() # case insensitive

        if reference == "epoch":
            return np.array([frame['epoch'] # WARNING, OLD VERSIONS USED SECONDS NOT NANOSECONDS 
                for frame in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=frames))
            ])
        
        if reference == "experiment":
            return np.array([frame['frameTimestamps_sec']
                for frame in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=frames))
            ])
        else:
            ValueError("Reference argument not a valid parameter (must be 'epoch' or 'experiment')")

    def find_events(self) -> list[siffutils.FrameMetaData]:
        """
        Returns a list of metadata objects corresponding to all frames where
        'events' occured, i.e. in which the Appended Text field is not empty.
        """
        list_of_list_of_events = [parseMetaAsEvents(meta) for meta in self.get_frames_metadata() if meta.hasEventTag]
        return [event for list_of_events in list_of_list_of_events for event in list_of_events]

    def epoch_to_frame_time(self, epoch_time : int) -> float:
        """ Converts epoch time to frame time for this experiment (returned in seconds) """

        meta = self.get_frames_metadata(frames=[0])
        frame_zero_time = meta[0]['frameTimestamps_sec'] # in seconds
        epoch_zero_time = meta[0]['epoch'] # in nanoseconds

        offset = frame_zero_time * SEC_TO_NANO - epoch_zero_time
        return (epoch_time + offset)/NANO_TO_SEC

### IMAGE INTENSITY METHODS
    def get_frames(self, frames: list[int] = None, flim : bool = False, 
        registration_dict : dict = None, discard_bins : int = None,
        ret_type : type = list
        ) -> list[np.ndarray]:
        """
        Returns the frames requested in frames keyword, or if None returns all frames.

        Wraps siffreader.get_frames

        INPUTS
        ------
        frames (optional) : list[int]

            Indices of input frames requested

        flim (optional) : bool

            Whether or not the returned np.ndarrays are 3d or 2d.

        registration_dict (optional) : dict

            Registration dictionary, if used

        discard_bins (optional) : int

            Arrival time bin beyond which to discard photons (if arrival times were measured).

        ret_type (optional) : type

            Type of returned PyObject. Default is list, if np.ndarray, will return an np.ndarray
            reshaped in order t, z, c, y, x, <tau> ("standard" order)

        RETURN VALUES
        -------------
        list[np.ndarray] 

            Each frame requested returned as a numpy array (either 2d or 3d).
        """
        if frames is None:
            frames = list(range(self.im_params.num_frames))
        if discard_bins is None:
            if registration_dict is None:
                framelist = siffreader.get_frames(frames = frames, flim = flim)
            else:
                framelist = siffreader.get_frames(frames = frames, flim = flim, registration = registration_dict)
        else:
            # arg checking
            if not isinstance(discard_bins, int):
                framelist = self.get_frames(frames, flim, registration_dict)
            else:
                if registration_dict is None:
                    framelist = siffreader.get_frames(frames = frames, flim = flim, discard_bins = discard_bins)
                else:
                    framelist = siffreader.get_frames(frames = frames, flim = flim, 
                                                registration = registration_dict, 
                                                discard_bins = discard_bins)

        if ret_type == list:
            return framelist

        if ret_type == np.ndarray:
            if self.im_params.frames_per_slice > 1:
                raise NotImplementedError(
                    "Array reshape hasn't been implemented if frames per slice > 1" +
                    "\nHaven't decided how to deal with the non-C-or-Fortan-style ordering yet."
                    )

            raise NotImplementedError("Haven't decided how to work out which slices / colors etc should be included")
            #stackshape = list(self.im_params.array_shape())

            #stackshape[0] = len(frames)/(stackshape[1]*stackshape[2])
            #return np.array(framelist).reshape(tuple(stackshape))

        raise ValueError(f"Invalid ret_type argument {ret_type}")

    def sum_across_time(self, timespan : int = 1, rolling : bool = False,
        timepoint_start : int = 0, timepoint_end : int = None,
        z_list : list[int] = None, color_list : list[int] = None,
        flim : bool = False, ret_type : type = list, registration_dict : dict = None,
        discard_bins : int = None, masks : list[np.ndarray] = None
        ) -> np.ndarray:
        """
        Sums adjacent frames in time of width "timespan" and returns a
        list of numpy arrays in standard form (or a single numpy array).

        TODO: IMPLEMENT RET_TYPE IN "STANDARD ORDER"

        INPUTS
        ------
        timespan (optional, default = 1) : int

            The number of TIMEPOINTS you want to pool together. This will determine the
            size of your output list: (timepoint_end - timepoint_start) / timespan.

        rolling (optional, default = False) : bool

            Take a rolling sum rather than a binned sum. Larger returned array size and longer time to compute.

        timepoint_start (optional, default = 0) : int

            The TIMEPOINT of the first frame you want to sum, so if you have a stack
            or multiple colors, this will not be the same as the frame number. If no
            input is given, this starts from the first timepoint.

        timepoint_end (optional, default = None) : int
        
            The TIMEPOINT of the last frame you want to sum, so if you have a stack
            or multiple colors, this will not be the same as the frame number. If no
            input is given, this will be the last timpeoint in the file.

        z_list (optional, default = None) : list[int]

            List of the z values you want to sum. If no list is given, defaults to the full volume.

        color_list (optional, default = None) : list[int]

            List of the color channels you want to sum. If no list is given, defaults to all present colors.

        flim (optional, default = False) : bool

            Whether to return FLIM arrays (y by x by tau) or INTENSITY arrays (y by x). Default is intensity.

        ret_type (optional, default = list) : type

            Determines the return type (either a single numpy array or a list of numpy arrays). The default
            option is list, but if you want a numpy array, input numpy.ndarray

        registration (optional): dict

            Registration dict for each frame

        discard_bins (optional): int

            Arrival times, in units of BINS, above which to discard photons

        masks (optional) : list[np.ndarray]

            List of numpy arrays to mask the returned array(s) with. If len(masks) > 1,
            then rather than returning a list, will return a list of lists. Each element
            of that list corresponds to the time series of the SUMMED input in each of the
            masks provided. So the format would be:

                [
                    [sum_of_mask_1_frame_1, sum_of_mask_1_frame_2, ... ],
                    [sum_of_mask_2_frame_1, sum_of_mask_2_frame_2, ...],
                    ...
                ]                


        RETURN VALUES
        -------------

        list or np.ndarray

            If list, returns a list of length len(color_list)*(timepoint_end-time_point_start)/timespan
            corresponding to the summed values of the pixels (or time bins) of TIMESPAN number of 
            successive timepoints.

            np.ndarray returns not yet implemented

        """

        ##### pre=processing
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")
        
        num_slices = self.im_params['NUM_SLICES']
        
        num_colors = 1
        if isinstance(self.im_params['COLORS'], list):
            num_colors = len(self.im_params['COLORS'])

        (z_list, flim, color_list) = x_across_time_TYPECHECK(num_slices, z_list, flim, num_colors, color_list)

        fps = self.im_params['FRAMES_PER_SLICE']

        timestep_size = int(self.im_params.frames_per_volume) # how many frames constitute a volume timepoint

        # figure out the start and stop points we're analyzing.
        frame_start = int(timepoint_start * timestep_size)
        
        if timepoint_end is None:
            frame_end = self.im_params['NUM_FRAMES'] - self.im_params.num_frames%self.im_params.frames_per_volume # only goes up to full volumes
        else:
            if timepoint_end > self.im_params['NUM_FRAMES']//timestep_size:
                logging.warning(
                    "\ntimepoint_end greater than total number of frames.\n"+
                    "Using maximum number of complete timepoints in image instead.\n"
                )
                timepoint_end = self.im_params['NUM_FRAMES']//timestep_size
            
            frame_end = timepoint_end*timestep_size 
            frame_end -= frame_end%self.im_params.frames_per_volume # subtract off non-full-volumes

        frame_end = int(frame_end) # frame_end is going to be the frame AFTER the last frame we actually DO include

        ##### the real stuff

        # now convert to a list for each set of frames we want to pool
        #
        # list comprehension makes this... incomprehensible. So let's do it
        # the generic way.
        framelist = []
        # a list for every element of a volume
        probe_lists = [[] for idx in range(timestep_size)]

        # offsets from the frame start that we actually want, as specified by
        # z_list and color_list
        viable_indices = [z*num_colors*fps + k*num_colors + c for z in z_list for k in range(fps) for c in color_list]

        if rolling:
            for volume_start in range(frame_start,frame_end, timestep_size):
                for vol_idx in range(timestep_size):
                    if vol_idx in viable_indices: # ignore undesired frames
                        framenum = vol_idx + volume_start # current frame
                        framelist.append(list(range(framenum, framenum+timestep_size*timespan, timespan)))
        if not rolling:
            # This approach is dumb and not super clever. Sure there's a better way.
            selected_frames_by_offset = []
            for offset_idx in viable_indices:
                 # list all of frames to be imaged, organized by offset_idx
                target_frames = list(range(frame_start + offset_idx, frame_end, timestep_size))
                selected_frames_by_offset.append([target_frames[idx:idx+timespan] for idx in range(0,len(target_frames),timespan)])
            framelist = list(itertools.chain(*list(zip(*selected_frames_by_offset)))) # ugly

        if masks is None:
            # ordered by time changing slowest, then z, then color, e.g.
            # T0: z0c0, z0c1, z1c0, z1c1, ... znc0, znc1, T1: ...
            if registration_dict is None:
                list_of_arrays = siffreader.pool_frames(framelist, flim=flim, discard_bins = discard_bins)
            else:
                list_of_arrays = siffreader.pool_frames(framelist, flim = flim, registration = registration_dict, discard_bins = discard_bins)

            if ret_type == np.ndarray:
                ## NOT YET IMPLEMENTED IN "STANDARD ORDER"
                return np.array(list_of_arrays)
            else:
                return list_of_arrays
        else:
            # TODO
            raise NotImplementedError("Using ROI masks in sum_across_time not yet implemented!")

    def pool_frames(self, 
            framelist : list[list[int]], 
            flim : bool = False,
            registration : dict = None,
            ret_type : type = list,
            discard_bins : int = None,
            masks : list[np.ndarray] = None 
        ) -> list[np.ndarray]:
        """
        Wraps siffreader.pool_frames
        TODO: Docstring.
        """
            
        if not masks is None:
            # TODO
            raise NotImplementedError("Haven't implemented masks in pool_frames yet")
        # ordered by time changing slowest, then z, then color, e.g.
        # T0: z0c0, z0c1, z1c0, z1c1, ... znz0, znc1, T1: ...
        if discard_bins is None:
            if registration is None:
               list_of_arrays = siffreader.pool_frames(framelist, flim=flim) 
            else:
                list_of_arrays = siffreader.pool_frames(framelist, flim=flim, registration= registration)
        else:
            if not isinstance(discard_bins, int):
                if registration is None:
                    list_of_arrays = siffreader.pool_frames(framelist, flim=flim)
                else:
                    return self.pool_frames(framelist, flim, registration, ret_type)
            else:
                if registration is None:
                    list_of_arrays = siffreader.pool_frames(framelist, flim=flim, discard_bins = discard_bins)
                else:
                    list_of_arrays = siffreader.pool_frames(framelist, flim=flim, 
                        registration = registration, discard_bins = discard_bins 
                    )

        if ret_type == list:
            return list_of_arrays

        if ret_type == np.ndarray:
            ## NOT YET IMPLEMENTED
            raise Exception("NDARRAY-TYPE RETURN NOT YET IMPLEMENTED.")

            frameshape = list_of_arrays[0].shape

            if flim:
                reshaped_dims = (-1, len(z_list),len(color_list),frameshape[0],frameshape[1],frameshape[2])
            else:
                reshaped_dims = (-1, len(z_list),len(color_list),frameshape[0],frameshape[1])
            
            np.array(list_of_arrays).reshape(reshaped_dims)

### FLIM METHODS
    def get_histogram(self, frames: list[int] = None) -> np.ndarray:
        """
        Get just the arrival times of photons in the list frames.

        Note: uses FRAME numbers, not timepoints. So you will mix color channels
        if you're not careful.
        
        INPUTS
        -----
        frames (optional, list of ints):

            Frames to get arrival times of. If NONE, collects from all frames.

        RETURN VALUES
        -------------
        histogram (np.ndarray):
            1 dimensional histogram of arrival times
        """
        if frames is None:
            return siffreader.get_histogram()
        return siffreader.get_histogram(frames=frames)[:self.im_params.num_bins]

    def histograms(self, color_channel : 'int|list' = None, frame_endpoints : tuple[int,int] = [None,None]) -> np.ndarray:
        """
        Returns a numpy array with arrival time histograms for all elements of the 
        keyword argument 'color_channel', which may be of type int or of type list (or any iterable).
        Each is stored on the major axis, so the returned array will be of dimensions:
        len(color_channel) x number_of_arrival_time_bins. You can define bounds in terms of numbers
        of frames (FOR THE COLOR CHANNEL, NOT TOTAL IMAGING FRAMES) with the other keyword argument
        frame_endpoints

        INPUTS
        ----------
        color_channel : int or list (default None)

            0 indexed list of color channels you want returned.
            If None is provided, returns for all color channels.

        frame_endpoints : tuple(int,int) (default (None, None))

            Start and end bounds on the frames from which to collect the histograms.
        """
        # I'm sure theres a more Pythonic way... I'm ignoring it
        # Accept the tuple, and then mutate it internal
        if type(frame_endpoints is tuple):
            frame_endpoints = list(frame_endpoints)
        if frame_endpoints[0] is None:
            frame_endpoints[0] = 0
        if frame_endpoints[1] is None:
            frame_endpoints[1] = int(self.im_params.num_volumes*self.im_params.frames_per_volume/self.im_params.num_colors)
        if color_channel is None:
            color_channel = [c-1 for c in self.im_params.colors]

        framelists = [framelist_by_color(self.im_params, c) for c in color_channel]
        true_framelists = [fl[frame_endpoints[0] : frame_endpoints[1]] for fl in framelists]
        return np.array([self.get_histogram(frames) for frames in true_framelists])

    def flimmap_across_time(self, flimfit : FLIMParams ,timespan : int = 1, rolling : bool = False,
            timepoint_start : int = 0, timepoint_end : int = None,
            z_list : list[int] = None, color_list : list[int] = None,
            ret_type : type = list, registration : dict = None,
            confidence_metric='chi_sq', discard_bins = None
        )-> np.ndarray:
        """
        Exactly as in sum_across_time but returns a FlimArray instead

        TODO: IMPLEMENT RET_TYPE
        TODO: IMPLEMENT NONE FOR CONFIDENCE METRIC

        INPUTS
        ------
        flimfit : FLIMParams

            The fit FLIM parameters for each color channel

        timespan (optional, default = 1) : int

            The number of TIMEPOINTS you want to pool together. This will determine the
            size of your output list: (timepoint_end - timepoint_start) / timespan.

        rolling (optional, default = False) : bool

            Whether or not to take a rolling sum, rather than binned. Larger output, longer compute time.

        timepoint_start (optional, default = 0) : int

            The TIMEPOINT of the first frame you want to sum, so if you have a stack
            or multiple colors, this will not be the same as the frame number. If no
            input is given, this starts from the first timepoint.

        timepoint_end (optional, default = None) : int
        
            The TIMEPOINT of the last frame you want to sum, so if you have a stack
            or multiple colors, this will not be the same as the frame number. If no
            input is given, this will be the last timpeoint in the file.

        z_list (optional, default = None) : list[int]

            List of the z values you want to sum. If no list is given, defaults to the full volume.

        color_list (optional, default = None) : list[int]

            List of the color channels you want to sum. If no list is given, defaults to all present colors.

        ret_type (optional, default = list) : type

            Determines the return type (either a single numpy array or a list of numpy arrays). The default
            option is list, but if you want a numpy array, input numpy.ndarray

        registration (optional, dict):

            Registration dict for each frame

        confidence_metric (optional, default='chi_sq') : str

            What metric to use to compute the confidence matrix returned in each tuple. Options:
                'None' Don't use it, tuple returned is length two (ARG MUST BE A STRING) TODO: do this right
                'chi_sq' Chi-squared statistic
                'log_p' log-likelihood of the pixel distribution

        discard_bins (optional) : int

            Arrival time, in units of BINS, above which to ignore photons


        RETURN VALUES
        -------------

        list or np.ndarray

            If list, returns a list of length len(color_list)*(timepoint_end-time_point_start)/timespan
            corresponding to the summed values of the pixels (or time bins) of TIMESPAN number of 
            successive timepoints.

            np.ndarray returns not yet implemented

        """

        ##### pre=processing
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")
            
        if not isinstance(flimfit, FLIMParams):
            raise TypeError("Argument flimfit not of type FLIMParams")
        
    
        num_slices = self.im_params['NUM_SLICES']
        
        num_colors = 1
        if isinstance(self.im_params['COLORS'], list):
            num_colors = len(self.im_params['COLORS'])

        (z_list, _, color_list) = x_across_time_TYPECHECK(num_slices, z_list, None, num_colors, color_list)

        fps = self.im_params['FRAMES_PER_SLICE']

        timestep_size = int(self.im_params.frames_per_volume) # how many frames constitute a volume timepoint

        # figure out the start and stop points we're analyzing.
        frame_start = int(timepoint_start * timestep_size)
        
        if timepoint_end is None:
            frame_end = self.im_params['NUM_FRAMES'] - self.im_params.num_frames%self.im_params.frames_per_volume # only goes up to full volumes
        else:
            if timepoint_end > self.im_params['NUM_FRAMES']//timestep_size:
                logging.warning(
                    "\ntimepoint_end greater than total number of frames.\n"+
                    "Using maximum number of complete timepoints in image instead.\n"
                )
                timepoint_end = self.im_params['NUM_FRAMES']//timestep_size
            
            frame_end = timepoint_end*timestep_size 
            frame_end -= frame_end%self.im_params.frames_per_volume # subtract off non-full-volumes

        frame_end = int(frame_end) # frame_end is going to be the frame AFTER the last frame we actually DO include

        ##### the real stuff

        # now convert to a list for each set of frames we want to pool
        #
        # list comprehension makes this... incomprehensible. So let's do it
        # the generic way.
        framelist = []
        # a list for every element of a volume
        probe_lists = [[] for idx in range(timestep_size)]

        # offsets from the frame start that we actually want, as specified by
        # z_list and color_list
        viable_indices = [z*num_colors*fps + k*num_colors + c for z in z_list for k in range(fps) for c in color_list]

        if rolling:
            for volume_start in range(frame_start,frame_end, timestep_size):
                for vol_idx in range(timestep_size):
                    if vol_idx in viable_indices: # ignore undesired frames
                        framenum = vol_idx + volume_start # current frame
                        framelist.append(list(range(framenum, framenum+timestep_size*timespan, timespan)))
        if not rolling:
            # This approach is dumb and not super clever. Sure there's a better way.
            selected_frames_by_offset = []
            for offset_idx in viable_indices:
                 # list all of frames to be imaged, organized by offset_idx
                target_frames = list(range(frame_start + offset_idx, frame_end, timestep_size))
                selected_frames_by_offset.append([target_frames[idx:idx+timespan] for idx in range(0,len(target_frames),timespan)])
            framelist = list(itertools.chain(*list(zip(*selected_frames_by_offset)))) # ugly
        
        # ordered by time changing slowest, then z, then color, e.g.
        # T0: z0c0, z0c1, z1c0, z1c1, ... znz0, znc1, T1: ...
        if registration is None:
            list_of_arrays = siffreader.flim_map(flimfit, frames = framelist, 
                                                confidence_metric=confidence_metric
                                                )
        else:
            list_of_arrays = siffreader.flim_map(flimfit, frames = framelist, registration = registration,
                                                confidence_metric=confidence_metric)

        if ret_type == np.ndarray:
            ## NOT YET IMPLEMENTED IN "STANDARD ORDER"
            return np.array(list_of_arrays)
        else:
            return [FlimArray(arr[1], arr[0], FLIMParams=flimfit) for arr in list_of_arrays]

### ORGANIZATIONAL METHODS
    def framelist_by_slice(self, color_channel : int = None) -> list[list[int]]:
        """
        Return a list of lists of the frames in the image corresponding to each z slice (but only one color channel)

        INPUTS
        ------

        color_channel : int

            Color channel to use for alignment (0-indexed). Defaults to 0, the green channel, if present.

        RETURN
        ------

        list[list[int]] :

            Returns a list of lists of ints, each sublist corresponding all frames across time for one z slice
            and one color channel.
        """
        
        return siffutils.framelist_by_slice(self.im_params, color_channel)
        
    def framelist_by_time(self, color_channel : int = None)->list[list[int]]:
        """
        Returns a list of lists of frame indices that are simultaneous (within a single color channel)

        INPUTS
        ------

        color_channel : int

            Color channel to use for alignment (0-indexed). Defaults to 0, the green channel, if present.

        RETURN
        ------

        list[list[int]] :

            Returns a list of lists of ints, each sublist corresponding all frames across z for one timepoint
            and one color channel.
        """
        
        return siffutils.framelist_by_timepoint(self.im_params, color_channel)

    def frames_to_single_array(self, frames=None):
        """
        TODO: IMPLEMENT
        Retrieves the frames in the list frames and uses the information retrieved from the header
        to parse them into an appropriately shaped (i.e. "standard order" tczyxtau) single array,
        rather than returning a list of numpy arrays

        INPUTS
        ------
        frames (array-like): list or array of the frame numbers to pool. If none, returns the full file.
        """
        raise NotImplementedError()

    def map_to_standard_order(self, numpy_array, map_list=['time','z','color','y','x','tau']):
        """
        TODO: IMPLEMENT
        Takes the numpy array numpy_array and returns it in the order (time, color, z, y, x, tau).
        Input arrays of dimension < 6 will be returned as 6 dimensional arrays with singleton dimensions.

        INPUTS
        ----------
        numpy_array: (ndarray)

        map_list: (list) List of any subset of the strings:
            "time"
            "z"
            "color"
            "y"
            "x"
            "tau"
            to make it clear which indices correspond to which dimension.
            If the input array has fewer dimensions than 6, that's fine.

        RETURN VALUES
        ----------
        reshaped: (ndarray) numpy_array reordered as the standard order, (time,z, color, y, x, tau)
        """
        raise NotImplementedError()

### REGISTRATION METHODS
    def register(self, reference_method="suite2p", color_channel : int = None, save : bool = True, 
        align_zplanes : bool = False, elastic_slice : float = np.nan, save_dict_name : str = None, **kwargs) -> dict:
        """
        Performs image registration by finding the rigid shift of each frame that maximizes its
        phase correlation with a reference image. The reference image is constructed according to 
        the requested reference method. If there are multiple color channels, defaults to the 'green'
        channel unless color_channel is passed as an argument. Returns the registration dict but also
        stores it in the Siffreader class. Internal alignment methods can be reached in the siffutils
        package, under the registration submodule. Aligns each z plane independently.

        Takes additional keyword args of siffutils.registration.register_frames
        
        TODO: provide support for more sophisticated registration methods

        INPUTS
        ------

        reference_method : str

            What method to use to construct the reference image for alignment. Options are:
                'average' -- Take the average over all frames in each z plane.
                'suite2p' -- Suite2p iterative alignment procedure. Much slower, much better.

        color_channel : int

            Color channel to use for alignment (0-indexed). Defaults to 0, the green channel, if present.

        save (optional) : bool

            Whether or not to save the dict. Name will be as TODO

        align_zplanes (optional) : bool

            Whether or not to align each z plane to the others.
        
        elastic_slice (optional) : float

            After each slice is registered, regularize estimated shifts. To ignore, use np.nan, None, or False.
            Defaults to off. The larger the argument, the stronger the "prior" (i.e. the less adjacent slices
            in time matter to compress the shift). Sometimes this works well. Sometimes it does not.

        save_dict_name (optional) : string
            
            What to name the saved pickled registration dict. Defaults to filename.dict

        Other kwargs are as in siffutils.registration.register_frames

        RETURN
        ------

        dict :

            A dict explaining the rigid transformation to apply to each frame. Form is:
                { FRAME_NUM_1 : (Y_SHIFT, X_SHIFT), FRAME_NUM_2 : (Y_SHIFT, X_SHIFT), ...}

                Can be passed directory to siffreader functionality that takes a registrationDict.
        """

        logging.warn("\n\n \t Don't forget to fix the zplane alignment!!")
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")

        if color_channel is None:
            color_channel = 0

        frames_list = self.framelist_by_slice(color_channel=color_channel)
        from .siffutils.registration import register_frames
        try:
            if __IPYTHON__: # check if we're running in a notebook. One of the nice things about an interpreted language!
                import tqdm
                pbar = tqdm.tqdm(frames_list)
                slicewise_reg =[register_frames(slice_element, ref_method=reference_method, tqdm = pbar, **kwargs) for slice_element in pbar]
            else:
                slicewise_reg =[register_frames(slice_element, ref_method=reference_method, **kwargs) for slice_element in frames_list]
        except (ImportError,NameError):
            slicewise_reg =[register_frames(slice_element, ref_method=reference_method, **kwargs) for slice_element in frames_list]
        
        # each element of the above is of the form:
        #   [registration_dict, difference_between_alignment_movements, reference image]

        # all we have to do now is do diagnostics to make sure there weren't any badly aligned frames
        # and then stick everything together into one dict
        
        # merge the dicts
        from functools import reduce
        reg_dict = reduce(lambda a, b : {**a, **b}, [slicewise_reg[n][0] for n in range(len(slicewise_reg))])

        # Decide if simultaneous frames from separate planes will be used to help align one another
        if not isinstance(elastic_slice,float):
            elastic_slice = 0.0
        if elastic_slice > 0.0:
            if np.abs(elastic_slice - np.sqrt(self.im_params['NUM_SLICES']-3)) < 0.3:
                logging.warning("\n\nELASTIC SLICE REGULARIZATION IS SINGULAR WHEN PARAMETER IS NEAR SQRT(N_SLICES-3)"
                              f"\nYOU USED {elastic_slice}"
                              f"\nDEFAULTING TO {elastic_slice+1.0} INSTEAD\n"
                             )
                elastic_slice = elastic_slice+1.0
            simultaneous_frames = self.framelist_by_time(color_channel=color_channel)
            from .siffutils.registration import regularize_all_tuples
            for framelist in simultaneous_frames:
                regularized = regularize_all_tuples(
                    [reg_dict[frame] for frame in framelist],
                    self.im_params['YSIZE'],
                    self.im_params['XSIZE'],
                    elastic_slice
                )
                for frame_idx in range(len(framelist)):
                    reg_dict[framelist[frame_idx]] = regularized[frame_idx]                    

        # Now align the reference frames so that all z planes are consistent, and shift all
        # planes by that amount
        if align_zplanes:
            logging.warn("Using the new align-stack-reference-frames feature. Double check to be sure it didn't mess things up!")

            # Hidden keyword argument:
            # ignore_first can be invoked here
            if 'ignore_first' in kwargs:
                refshift = registration.align_references([reg_tuple[2] for reg_tuple in slicewise_reg], ignore_first = bool(kwargs['ignore_first']))
            else:
                refshift = registration.align_references([reg_tuple[2] for reg_tuple in slicewise_reg])
            
            slicewise = self.framelist_by_slice()
            for z in range(len(slicewise)):
                slicewise_reg[z] = (
                    slicewise_reg[z][0],
                    slicewise_reg[z][1],
                    np.roll(slicewise_reg[z][2],refshift[z]) # roll the reference frames before they're saved
                )
                for frame in slicewise[z]:
                    reg_dict[frame] = ( # shift all the registration tuples
                            int((reg_dict[frame][0] + refshift[z][0])%self.im_params.ysize),
                            int((reg_dict[frame][1] + refshift[z][1])%self.im_params.xsize)
                        )


        # Apply to contemporaneous color channels if there are some
        if isinstance(self.im_params['COLORS'], list):
            # the color-1 is to 0-index
            # if color_channel is 1, offsets is [-1, 0]
            # if color_channel is 0, offsets is [0, 1]
            offsets = [(color-1) - color_channel for color in self.im_params['COLORS']]
            keylist = list(reg_dict.keys())
            
            for key in keylist:
                for offset in offsets:
                    reg_dict[key + offset] = reg_dict[key]

        # Now store the registration dict
        self.registration_dict = reg_dict
        self.reference_frames = [reg_tuple[2] for reg_tuple in slicewise_reg]

        if save:
            if save_dict_name is None:
                save_dict_name = os.path.splitext(self.filename)[0] 
            with open(save_dict_name + ".dict",'wb') as dict_path:
                pickle.dump(reg_dict, dict_path)
            with open(save_dict_name + ".ref",'wb') as ref_images_path:
                pickle.dump(self.reference_frames, ref_images_path)

        return reg_dict

    def align_reference_frames(self, ignore_first : bool = True, overwrite = True, **kwargs):
        """
        Used to align reference frames (and shift the registration dictionary)
        after registration has already been performed without the align_zplanes
        option. Overwrites saved dicts if they exist by default, can be adjusted
        with the parameter overwrite.

        Arguments
        ---------
        
        ignore_first : bool (optional)

            Whether or not to ignore the first z plane during alignment. Usually this contains
            a lot of artifacts due to the piezo flyback, so it's set to True by default

        overwrite : bool (optional)

            Whether or not overwrite any existing registration dictionary.
        """
        raise NotImplementedError()

### DEBUG METHODS   
    def set_debug(self, debug : bool):
        """ 
        Toggles debug features of the SiffReader class on and off.
        """
        self.debug = debug

#########
#########
# LOCAL #
#########
#########

def suppress_warnings() -> None:
    siffreader.suppress_warnings()

def report_warnings() -> None:
    siffreader.report_warnings()

# I don't think this is a natural home for these functions but they rely on multiple siffutils and so they end up with
# circular imports if I put them in most of those places... To figure out later. TODO
def channel_exp_fit(photon_arrivals : np.ndarray, num_components : int = 2, initial_fit : dict = None, color_channel : int = None, **kwargs) -> FLIMParams:
    """
    Takes row data of arrival times and returns the param dict from an exponential fit.
    TODO: Provide more options to how fitting is done


    INPUTS
    ----------

    photon_arrivals (1-dim ndarray): Histogrammed arrival time of each photon.

    num_components (int): Number of components to the exponential TODO: enable more diversity?

    initial_fit (dict): FLIMParams formatted dict of first-guess FLIM fit.


    RETURN VALUES
    ----------
    FLIMParams -- (FLIMParams object)
    """
    if (initial_fit is None):
        if num_components == 2:
            initial_fit = { # pretty decent guess for Camui data
                'NCOMPONENTS' : 2,
                'EXPPARAMS' : [
                    {'FRAC' : 0.7, 'TAU' : 115},
                    {'FRAC' : 0.3, 'TAU' : 25}
                ],
                'CHISQ' : 0.0,
                'T_O' : 20,
                'IRF' :
                    {
                        'DIST' : 'GAUSSIAN',
                        'PARAMS' : {
                            'SIGMA' : 4
                        }
                    }
            }
        if num_components == 1:
            initial_fit = { # GCaMP / free GFP fluoroscence
                'NCOMPONENTS' : 1,
                'EXPPARAMS' : [
                    {'FRAC' : 1.0, 'TAU' : 140}
                ],
                'CHISQ' : 0.0,
                'T_O' : 20,
                'IRF' :
                    {
                        'DIST' : 'GAUSSIAN',
                        'PARAMS' : {
                            'SIGMA' : 4
                        }
                    }
            }


    params = FLIMParams(param_dict=initial_fit, color_channel = color_channel, **kwargs)
    params.fit_data(photon_arrivals,num_components=num_components, x0=params.param_tuple())
    if not params.CHI_SQD > 0:
        logging.warn("Returned FLIMParams object has a chi-squared of zero, check the fit to be sure!")
    return params

def fit_exp(histograms : np.ndarray, num_components: 'int|list[int]' = 2, fluorophores : list[str] = None, use_noise : bool = False) -> list[FLIMParams]:
    """
    params = siffpy.fit_exp(histograms, num_components=2)


    Takes a numpy array with dimensions n_colors x num_arrival_time_bins
    returns a color-element list of dicts with fits of the fluorescence emission model for each color channel

    INPUTS
    ------
    histograms: (ndarray) An array of data with the following shapes:
        (num_bins,)
        (1, num_bins)
        (n_colors, num_bins)

    num_components: 
    
        (int) Number of exponentials in the fit (one color channel)
        (list) Number of exponentials in each fit (per color channel)

        If there are more color channels than provided in this list (or it's an
        int with a multidimensional histograms input), they will be filled with
        the value 2.

    fluorophores (list[str] or str):

        List of fluorophores, in same order as color channels. By default, is None.
        Used for initial conditions in fitting the exponentials. I doubt it's critical.

    use_noise (bool, optional):

        Whether or not to put noise in the FLIMParameter fit by default
    
    RETURN VALUES
    -------------

    fits (list):
        A list of FLIMParams objects, containing fit parameters as attributes and with functionality
        to return the parameters as a tuple or dict.

    """

    n_colors = 1
    if len(histograms.shape)>1:
        n_colors = histograms.shape[0]

    if n_colors > 1:
        if not (type(num_components) == list):
            num_components = [num_components]
        if len(num_components) < n_colors:
            num_components += [2]*(n_colors - len(num_components)) # fill with 2s

    # type-checking -- HEY I thought this was Python!
    if not (isinstance(fluorophores, list) or isinstance(fluorophores, str)):
        fluorophores = None

    # take care of the fluorophores arg
    if fluorophores is None:
        fluorophores = [None] * n_colors

    if len(fluorophores) < n_colors:
        fluorophores += [None] * (n_colors - len(fluorophores)) # pad with Nones

    # take these strings, turn them into initial guesses for the fit parameters
    availables = siffutils.available_fluorophores(dtype=dict)

    for idx in range(len(fluorophores)):
        if not (fluorophores[idx] in availables):
            logging.warning("\n\nProposed fluorophore %s not in known fluorophore list. Using default params instead\n" % (fluorophores[idx]))
            fluorophores[idx] = None

    list_of_dicts_of_fluorophores = [availables[tool_name] if isinstance(tool_name,str) else None for tool_name in fluorophores]
    list_of_flimparams = [FLIMParams(param_dict = this_dict, use_noise = use_noise) if isinstance(this_dict, dict) else None for this_dict in list_of_dicts_of_fluorophores]
    fluorophores_dict_list = [FlimP.param_dict() if isinstance(FlimP, FLIMParams) else None for FlimP in list_of_flimparams]

    if n_colors>1:
        fit_list = [channel_exp_fit( histograms[x,:],num_components[x], fluorophores_dict_list[x] , color_channel = x, use_noise = use_noise) for x in range(n_colors)]
    else:
        fit_list = [channel_exp_fit( histograms,num_components, fluorophores_dict_list[0], use_noise = use_noise )]

    return fit_list