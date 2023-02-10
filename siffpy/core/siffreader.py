from typing import Union
import itertools
import logging, os, pickle
from functools import reduce
from pathlib import Path
import builtins

import numpy as np

from siffreadermodule import SiffIO

from siffpy.core import io, timetools
from siffpy.core.flim import FLIMParams, FlimUnits
from siffpy.core.utils import ImParams, registration
from siffpy.core.utils.typecheck import *
from siffpy.core.utils.registration import register_frames, regularize_all_tuples
from siffpy.siffmath.flim import FlimTrace

# TODO:
# __repr__

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
    def __init__(self, filename : Union[str, Path] = None):
        self.im_params : ImParams = None
        self.ROI_group_data = {}
        self.opened = False
        self.registration_dict = None
        self.reference_frames = None
        self.debug = False
        self.events = None
        self.siffio = SiffIO()
        if isinstance(filename, Path):
            filename = str(filename)
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
        """
        Prints a pretty representation of the SiffReader
        where the open file, if any, is listed first, followed by the image parameters,
        whether a registration dictionary is loaded, and whether reference images are loaded.
        
        """
        # TODO
        return self.__str__()

    def open(self,  filename : Union[str, Path] = None) -> None:
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
        if isinstance(filename, Path):            
            filename = str(filename)            
        if self.opened and not (filename == self.filename):
            self.siffio.close()
        
        print(f"Opening {filename}, collecting metadata...")
        self.siffio.open(filename)
        self.filename = filename

        header = self.siffio.get_file_header()
        if self.debug:
            print("Header read")

        self.im_params = io.header_to_imparams(header, self.siffio.num_frames())

        self.ROI_group_data = io.header_data_to_roi_string(header)
        self.opened = True

        # TODO: PUT THIS SOMEWHERE LESS CLUMSY
        if os.path.exists(os.path.splitext(filename)[0] + ".dict"):
            with open(os.path.splitext(filename)[0] + ".dict", 'rb') as dict_file:
                reg_dict = pickle.load(dict_file)
            if isinstance(reg_dict, dict):
                print("\n\n\tFound a registration dictionary for this image and importing it.\n")
                self.registration_dict = reg_dict
            else:
                logging.warning("\n\n\tPutative registration dict for this file is not of type dict.\n")
        if os.path.exists(os.path.splitext(filename)[0] + ".ref"):
            with open(os.path.splitext(filename)[0] + ".ref", 'rb') as images_list:
                ref_ims = pickle.load(images_list)
            if isinstance(ref_ims, list):
                print("\n\n\tFound a reference image list for this file and importing it.\n")
                self.reference_frames = ref_ims
            else:
                logging.warning("\n\n\tPutative reference images object for this file is not of type list.\n", stacklevel=2)
        
        self.events = io.find_events(self.im_params, self.get_frames_metadata())

    def close(self) -> None:
        """ Closes opened file """
        self.siffio.close()
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
    def t_axis(self,
        timepoint_start : int = 0,
        timepoint_end : int = None,
        reference_z : int = 0,
        reference_time : str = 'experiment'
    ) -> np.ndarray:
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
        
        reference_time (optional, str):
            Referenced to start of the experiment, or referenced to epoch time.

            Possible values:
                experiment - referenced to experiment
                epoch      - referenced to epoch

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
        
        framelist = self.im_params.flatten_by_timepoints(
            timepoint_start,
            timepoint_end,
            reference_z
        )

        return timetools.metadata_dicts_to_time(
            self.siffio.get_frame_metadata(frames=framelist),
            reference = reference_time,
        )
    
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
            frames = list(range(self.im_params.num_frames))

        return timetools.metadata_dicts_to_time(self.siffio.get_frame_metadata(frames=frames), reference)

### METADATA METHODS
    def get_frames_metadata(self, frames : list[int] = None) -> list[io.FrameMetaData]:
        if frames is None:
            frames = list(range(self.im_params.num_frames))
        return [io.FrameMetaData(meta_dict)
            for meta_dict in io.frame_metadata_to_dict(self.siffio.get_frame_metadata(frames=frames))
        ]

    def epoch_to_frame_time(self, epoch_time : int) -> float:
        """ Converts epoch time to frame time for this experiment (returned in seconds) """
        return timetools.epoch_to_frame_time(epoch_time, self.get_frames_metadata(frames = [0])[0])
        

### IMAGE INTENSITY METHODS
    def get_frames(self,
        frames: list[int] = None,
        flim : bool = False, 
        registration_dict : dict = None,
        discard_bins : int = None,
        ret_type : type = list
        ) -> list[np.ndarray]:
        """
        Returns the frames requested in frames keyword, or if None returns all frames.

        Wraps self.siffio.get_frames

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
                framelist = self.siffio.get_frames(frames = frames, flim = flim)
            else:
                framelist = self.siffio.get_frames(frames = frames, flim = flim, registration = registration_dict)
        else:
            # arg checking
            if not isinstance(discard_bins, int):
                framelist = self.siffio.get_frames(frames, flim, registration_dict)
            else:
                if registration_dict is None:
                    framelist = self.siffio.get_frames(frames = frames, flim = flim, discard_bins = discard_bins)
                else:
                    framelist = self.siffio.get_frames(frames = frames, flim = flim, 
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

        raise ValueError(f"Invalid ret_type argument {ret_type}")

    def sum_mask(
            self,
            mask : np.ndarray,
            timepoint_start : int = 0,
            timepoint_end : int = None,
            z_list : list[int] = None,
            color_channel :  int = 1,
            registration_dict : dict = None,
            discard_bins : int = None,
        )->np.ndarray:
        """
        Computes the sum photon counts within a numpy mask over timesteps.
        Takes _timepoints_ as arguments, not frames.

        Arguments
        ---------
        mask : np.ndarray[bool]

            Mask to sum over. Must be the same shape as the image.

        timepoint_start : int
            
            Starting timepoint for the sum. Default is 0.

        timepoint_end : int

            Ending timepoint for the sum. Default is None, which means the last timepoint.

        z_list : list[int]

            List of z-slices to sum over. Default is None, which means all z-slices.

        color_channel : int

            Color channel to sum over. Default is 1, which means the FIRST color channel.
        
        registration_dict : dict

            Registration dictionary, if used

        discard_bins : int

            Arrival time bin beyond which to discard photons (if arrival times were measured).

        Returns
        -------
        np.ndarray

            Summed photon counts as an array of shape (n_timepoints, mask.shape[0])
        """

        if timepoint_end is None:
            timepoint_end = self.im_params.num_timepoints

        if z_list is None:
            z_list = list(range(self.im_params.num_slices))

        if isinstance(z_list, int):
            z_list = [z_list]

        if registration_dict is None and hasattr(self, 'registration_dict'):
            registration_dict = self.registration_dict

        if not (discard_bins is None):
            logging.warning("Discard_bins keyword argument not implemented, ignoring")

        frames = self.im_params.framelist_by_slices(color_channel = color_channel-1, slices = z_list, lower_bound=timepoint_start, upper_bound=timepoint_end)
        
        summed_data = self.siffio.sum_roi(
            mask,
            frames = frames,
            registration = registration_dict
        )

        # more than one slice, sum across slices
        return np.sum(summed_data.reshape((len(z_list),-1)),axis=0)
            
    def sum_roi(self, roi : 'siffpy.siffplot.roi_protocols.rois.ROI',
        timepoint_start : int = 0, timepoint_end : int = None,
        color_list : list[int] = None, registration_dict : dict = None,
        )->Union[np.ndarray,list[np.ndarray]]:
        """

        TODO: Rewrite to use sum_mask.

        Computes the sum photon counts within an ROI over timesteps.

        Color_list should be organized by CHANNELS not by the INDEX,
        so it's 1-indexed NOT 0-indexed.

        If color_list is a list, returns a list, which each element being a color
        channel.

        If color_list is an int, returns a numpy array.

        If color_list is None, returns all color channels as if a list were provided.
        If there's only one color, will return as a numpy array no matter what.
        """
        if color_list is None:
            color_list = self.im_params.color_list

        if self.im_params.num_colors == 1:
            color_list = 0

        if timepoint_end is None:
            timepoint_end = self.im_params.num_timepoints

        if registration_dict is None:
            if hasattr(self,'registration_dict'):
                registration_dict = self.registration_dict

        if isinstance(color_list, int):
            color = color_list-1
            slice_idx = None
            if hasattr(roi, 'slice_idx'):
                slice_idx = roi.slice_idx
            frames = self.im_params.framelist_by_slice(color_channel = color, slice_idx = slice_idx)
            if slice_idx is None: # flattens the list to extract values, then later will compress
            # along slices
                print(len(frames))
                frames = [individual_frame[timepoint_start:timepoint_end] for slicewise in frames for individual_frame in slicewise]
            else:
                frames = frames[timepoint_start:timepoint_end]
            
            try:
                mask = roi.mask()
            except ValueError:
                if slice_idx is None:
                    mask = roi.mask(image=self.reference_frames[0])
                else:
                    mask = roi.mask(image=self.reference_frames[slice_idx])

            summed_data = self.siffio.sum_roi(
                mask,
                frames = frames,
                registration = registration_dict
            )

            # just one slice, no need to repack
            if isinstance(slice_idx, int):
                return summed_data

            # more than one slice, sum across slices
            return np.sum(summed_data.reshape((-1, self.im_params.num_slices)),axis=-1)
            
        # This means color_list is iterable
        output_list = []
        for color in color_list:
            color = color-1
            # Do the above, but iterate through colors
            slice_idx = None
            if hasattr(roi, 'slice_idx'):
                slice_idx = roi.slice_idx
            frames = self.im_params.framelist_by_slice(color_channel = color, slice_idx = slice_idx)
            if slice_idx is None: # flattens the list to extract values, then later will compress
            # along slices
                frames = [individual_frame for slicewise in frames for individual_frame in slicewise]
            else:
                frames = frames[timepoint_start:timepoint_end]

            try:
                mask = roi.mask()
            except ValueError:
                if slice_idx is None:
                    mask = roi.mask(image=self.reference_frames[0])
                else:
                    mask = roi.mask(image=self.reference_frames[slice_idx])

            summed_data = self.siffio.sum_roi(
                mask,
                frames = frames,
                registration = registration_dict
            )

            if not isinstance(slice_idx,int):
                summed_data = np.sum(summed_data.reshape((-1, self.im_param.num_slices)),axis=-1)

            output_list.append(summed_data)

        return output_list

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
        
        num_slices = self.im_params.num_slices
        
        num_colors = self.im_params.num_colors

        (z_list, flim, color_list) = x_across_time_TYPECHECK(num_slices, z_list, flim, num_colors, color_list)

        fps = self.im_params.frames_per_slice

        timestep_size = int(self.im_params.frames_per_volume) # how many frames constitute a volume timepoint

        # figure out the start and stop points we're analyzing.
        frame_start = int(timepoint_start * timestep_size)
        
        if timepoint_end is None:
            frame_end = self.im_params.num_frames - self.im_params.num_frames%self.im_params.frames_per_volume # only goes up to full volumes
        else:
            if timepoint_end > self.im_params.num_frames//timestep_size:
                logging.warning(
                    "\ntimepoint_end greater than total number of frames.\n"+
                    "Using maximum number of complete timepoints in image instead.\n"
                )
                timepoint_end = self.im_params.num_frames//timestep_size
            
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
                list_of_arrays = self.siffio.pool_frames(framelist, flim=flim, discard_bins = discard_bins)
            else:
                list_of_arrays = self.siffio.pool_frames(framelist, flim = flim, registration = registration_dict, discard_bins = discard_bins)

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
        Wraps self.siffio.pool_frames
        TODO: Docstring.
        """
            
        if not masks is None:
            # TODO
            raise NotImplementedError("Haven't implemented masks in pool_frames yet")
        # ordered by time changing slowest, then z, then color, e.g.
        # T0: z0c0, z0c1, z1c0, z1c1, ... znz0, znc1, T1: ...
        if discard_bins is None:
            if registration is None:
               list_of_arrays = self.siffio.pool_frames(framelist, flim=flim) 
            else:
                list_of_arrays = self.siffio.pool_frames(framelist, flim=flim, registration= registration)
        else:
            if not isinstance(discard_bins, int):
                if registration is None:
                    list_of_arrays = self.siffio.pool_frames(framelist, flim=flim)
                else:
                    return self.pool_frames(framelist, flim, registration, ret_type)
            else:
                if registration is None:
                    list_of_arrays = self.siffio.pool_frames(framelist, flim=flim, discard_bins = discard_bins)
                else:
                    list_of_arrays = self.siffio.pool_frames(framelist, flim=flim, 
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
            return self.siffio.get_histogram()
        return self.siffio.get_histogram(frames=frames)[:self.im_params.num_bins]

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
        if isinstance(color_channel, int):
            color_channel = [color_channel]

        framelists = [self.im_params.framelist_by_color(c) for c in color_channel]
        true_framelists = [fl[frame_endpoints[0] : frame_endpoints[1]] for fl in framelists]
        return np.array([self.get_histogram(frames) for frames in true_framelists])

    def get_frames_flim(self,
            params : FLIMParams,
            frames: list[int] = None,
            registration_dict : dict = None,
            confidence_metric : str = 'chi_sq',
        ) -> FlimTrace:
        """
        Returns a FlimTrace object of dimensions
        n_frames by y_size by x_size corresponding
        to the frames requested.
        """
     
        if frames is None:
            frames = list(range(self.im_params.num_frames))
        
        if registration_dict is None:
            list_of_arrays = self.siffio.flim_map(
                params,
                frames = frames, 
                confidence_metric=confidence_metric
            )
        else:
            list_of_arrays = self.siffio.flim_map(
                params,
                frames = frames,
                registration = registration_dict,
                confidence_metric=confidence_metric
            )

        return FlimTrace(
            np.array(list_of_arrays[0]),
            intensity = np.array(list_of_arrays[1]),
            FLIMParams = params,
            method = 'empirical lifetime',
        )

    def sum_roi_flim(self,
            params : Union[FLIMParams, list[FLIMParams]],
            roi : 'siffpy.siffplot.roi_protocols.rois.ROI',
            timepoint_start : int = 0,
            timepoint_end : int = None,
            color_list : list[int] = None,
            registration_dict : dict = None,
        )->Union[FlimTrace,list[FlimTrace]]:
        """
        Computes the empirical lifetime within an ROI over timesteps.

        params determines the color channels used.

        If params is a list, returns a list of numpy arrays, each corresponding
        to the provided FLIMParams element.

        If params is a single FLIMParams object, returns a numpy array.

        ARGUMENTS
        ----------

        params : FLIMParams object or list of FLIMParams

            The FLIMParams objects fit to the FLIM data of this .siff file. If
            the FLIMParams objects do not all have a color_channel attribute,
            then the optional argument color_list must be provided and be a list
            of ints corresponding to the * 1-indexed * color channel numbers for
            each FLIMParams, unless there is only one color channel in the data.

        roi : siffpy.siffplot.roi_protocols.rois.ROI

            An ROI subclass that defines the boundaries of the region being considered.

        timepoint_start : int (optional) (default is 0)

            The TIMEPOINT (not frame) at which to start the analysis. Defaults to 0.

        timepoint_end : int (optional) (default is None)

            The TIMEPOINT (not frame) at which the analysis ends. If the argument is None,
            defaults to the final timepoint of the .siff file.

        color_list : list[int] (optional) (default is None)

            If the FLIMParams objects do not have a color_channel attribute, then this
            argument is REQUIRED. Explains which channel's frames should used to compute
            the empirical lifetime. Must be of the same length as the params argument. Will
            be superceded by the corresponding FLIMParams color_channel if both are required.

        registration_dict : dict

            Registration dictionary for frames.

        """

        if isinstance(params, FLIMParams):
            params = [params]

        if not isinstance(params, list):
            raise ValueError("params argument must be either a FLIMParams object or a list of FLIMParams")

        if not all(isinstance(x, FLIMParams) for x in params):
            raise ValueError("params argument must be either a FLIMParams object or a list of FLIMParams")

        # Make color_list a list no matter what so that it's mutable
        if hasattr(color_list, '__iter__'):
            color_list = [color for color in color_list]

        if self.im_params.num_colors == 1:
            color_list = [0]*len(params)
        
        # Now check that the color parameters are all correct
        if color_list is None:
            if not all(hasattr(x,'color_channel') and not (x.color_channel is None) for x in params):
                raise ValueError(
                    "Provided FLIMParams do not all have a valid color_channel, "
                    "more than one color channel is in the given siff file, "
                    "and no color_list argument was provided."
                )
            color_list = [None]*len(params)

        # not everything needs to be Pythonic. This just seems more readable
        for idx, param in enumerate(params):
            if hasattr(param, 'color_channel') and not (param.color_channel is None):
                color_list[idx] = int(param.color_channel)
        
        if not all(isinstance(x, int) for x in color_list):
            raise ValueError("At least one provided FLIMParams does not have a "
                "color channel that has been accounted for, either as a color_channel attribute "
                "or as an element of the color_list keyword argument."
            )        

        if timepoint_end is None:
            timepoint_end = self.im_params.num_timepoints

        if registration_dict is None:
            if hasattr(self,'registration_dict'):
                registration_dict = self.registration_dict

        if len(params) == 1:
            color = color_list[0]-1
            slice_idx = None
            if hasattr(roi, 'slice_idx'):
                slice_idx = roi.slice_idx
            frames = self.im_params.framelist_by_slice(color_channel = color, slice_idx = slice_idx)
            if slice_idx is None: # flattens the list to extract values, then later will compress
            # along slices
                frames = [individual_frame[timepoint_start:timepoint_end] for slicewise in frames for individual_frame in slicewise]
            else:
                frames = frames[timepoint_start:timepoint_end]
            try:
                mask = roi.mask()
            except ValueError:
                if slice_idx is None:
                    mask = roi.mask(image=self.reference_frames[0])
                else:
                    mask = roi.mask(image=self.reference_frames[slice_idx])

            summed_intensity_data = self.siffio.sum_roi(
                mask,
                frames = frames,
                registration = registration_dict
            )

            summed_flim_data = self.siffio.sum_roi_flim(
                mask,
                params[0],
                frames = frames,
                registration = registration_dict
            ) * self.im_params.picoseconds_per_bin

            # just one slice, no need to repack
            if isinstance(slice_idx, int):
                return FlimTrace(
                    summed_flim_data,
                    intensity = summed_intensity_data,
                    FLIMParams = params[0],
                    method = 'empirical lifetime',
                    info_string =  "ROI ID: " + roi.hashname,
                    units = FlimUnits.PICOSECONDS
                )

            # more than one slice, sum across slices TODO!!!!
            summed_intensity_data.reshape((-1, self.im_params.num_slices))
            summed_flim_data.reshape((-1, self.im_params.num_slices))
            return np.sum(
                FlimTrace( # a numpy operation on the list alone returns just a numpy array
                    [
                        FlimTrace(
                            summed_flim_data[...,k], 
                            intensity = summed_intensity_data[...,k],
                            FLIMParams = params[0],
                            method = 'empirical lifetime',
                            info_string = "ROI ID " + roi.hashname,
                            units = FlimUnits.PICOOSECONDS
                        )
                        for k in range(self.im_params.num_slices)
                    ]
                ),
                axis=-1
            )
            
        # This means color_list is longer than length 1
        output_list = []
        for idx, color in enumerate(color_list):
            color = color-1
            # Do the above, but iterate through colors
            slice_idx = None
            if hasattr(roi, 'slice_idx'):
                slice_idx = roi.slice_idx
            frames = self.im_params.framelist_by_slice(color_channel = color, slice_idx = slice_idx)
            if slice_idx is None: # flattens the list to extract values, then later will compress
            # along slices
                frames = [individual_frame for slicewise in frames for individual_frame in slicewise]
            else:
                frames = frames[timepoint_start:timepoint_end]

            try:
                mask = roi.mask()
            except ValueError:
                if slice_idx is None:
                    mask = roi.mask(image=self.reference_frames[0])
                else:
                    mask = roi.mask(image=self.reference_frames[slice_idx])

            summed_intensity_data = self.siffio.sum_roi(
                mask,
                frames = frames,
                registration = registration_dict
            )

            summed_flim_data = self.siffio.sum_roi_flim(
                mask,
                params[idx],
                frames = frames,
                registration = registration_dict
            ) * self.im_params.picoseconds_per_bin

            if not isinstance(slice_idx,int):
                raise ValueError("Haven't set up to handle more than one slice per ROI in this function yet.")
                summed_data = np.sum(summed_data.reshape((-1, self.im_param.num_slices)),axis=-1)

            output_list.append(
                FlimTrace(
                    summed_flim_data,
                    intensity = summed_intensity_data,
                    method = 'empirical lifetime',
                    FLIMParams = params[idx],
                    info_string = "ROI ID: " + roi.hashname,
                    units = FlimUnits.PICOSECONDS
                )
            )

        return FlimTrace(output_list)

    def flimmap_across_time(self, flimfit : FLIMParams ,timespan : int = 1, rolling : bool = False,
            timepoint_start : int = 0, timepoint_end : int = None,
            z_list : list[int] = None, color_list : list[int] = None,
            ret_type : type = list, registration : dict = None,
            confidence_metric='chi_sq', discard_bins = None
        )-> FlimTrace:
        """

        Exactly as in sum_across_time but returns a FlimArray instead

        TODO: IMPLEMENT RET_TYPE
        TODO: IMPLEMENT NONE FOR CONFIDENCE METRIC
        TODO: MORE PROPER DOCSTRING

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
        
        # MAKE SURE IT'S IN COUNTBINS, IF NOT MAKE A COPY
        #if not ()
    
        num_slices = self.im_params.num_slices
        
        num_colors = self.im_params.num_colors

        (z_list, _, color_list) = x_across_time_TYPECHECK(num_slices, z_list, None, num_colors, color_list)

        fps = self.im_params.frames_per_slice

        timestep_size = int(self.im_params.frames_per_volume) # how many frames constitute a volume timepoint

        # figure out the start and stop points we're analyzing.
        frame_start = int(timepoint_start * timestep_size)
        
        if timepoint_end is None:
            frame_end = self.im_params.num_frames - self.im_params.num_frames%self.im_params.frames_per_volume # only goes up to full volumes
        else:
            if timepoint_end > self.im_params.num_frames//timestep_size:
                logging.warning(
                    "\ntimepoint_end greater than total number of frames.\n"+
                    "Using maximum number of complete timepoints in image instead.\n"
                )
                timepoint_end = self.im_params.num_frames//timestep_size
            
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
            list_of_arrays = self.siffio.flim_map(flimfit, frames = framelist, 
                                                confidence_metric=confidence_metric
                                                )
        else:
            list_of_arrays = self.siffio.flim_map(flimfit, frames = framelist, registration = registration,
                                                confidence_metric=confidence_metric)

        if ret_type == np.ndarray:
            ## NOT YET IMPLEMENTED IN "STANDARD ORDER"
            return np.array(list_of_arrays)
        else:
            return FlimTrace(
                [
                    FlimTrace(
                        arr[0],
                        intensity = arr[1],
                        FLIMParams = flimfit,
                        method = 'empirical lifetime',
                        units = FlimUnits.COUNTBINS,
                    )
                    for arr in list_of_arrays
                ]
            )

### REGISTRATION METHODS
    def register(
        self,
        reference_method="suite2p",
        color_channel : int = None,
        save : bool = True, 
        align_zplanes : bool = False,
        elastic_slice : float = np.nan,
        save_dict_name : str = None,
        **kwargs
        ) -> dict:
        """
        Performs image registration by finding the rigid shift of each frame that maximizes its
        phase correlation with a reference image. The reference image is constructed according to 
        the requested reference method. If there are multiple color channels, defaults to the 'green'
        channel unless color_channel is passed as an argument. Returns the registration dict but also
        stores it in the Siffreader class. Internal alignment methods can be reached in the registration
        program registration.py in siffpy.core.utils. Aligns each z plane independently.

        Takes additional keyword args of siffpy.core.utils.registration.register_frames
        
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

        Other kwargs are as in siffpy.core.utils.registration.register_frames

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

        frames_list = self.im_params.framelist_by_slice(color_channel=color_channel)

        try:
            if hasattr(builtins, "__IPYTHON__"): # check if we're running in a notebook. One of the nice things about an interpreted language!
                import tqdm
                pbar = tqdm.tqdm(frames_list)
                slicewise_reg =[register_frames(self.siffio, slice_element, ref_method=reference_method, tqdm = pbar, **kwargs) for slice_element in pbar]
            else:
                slicewise_reg =[register_frames(self.siffio, slice_element, ref_method=reference_method, **kwargs) for slice_element in frames_list]
        except (ImportError,NameError):
            slicewise_reg =[register_frames(self.siffio, slice_element, ref_method=reference_method, **kwargs) for slice_element in frames_list]
        
        # each element of the above is of the form:
        #   [registration_dict, difference_between_alignment_movements, reference image]

        # all we have to do now is do diagnostics to make sure there weren't any badly aligned frames
        # and then stick everything together into one dict
        
        # merge the dicts
        reg_dict = reduce(lambda a, b : {**a, **b}, [slicewise_reg[n][0] for n in range(len(slicewise_reg))])

        # Decide if simultaneous frames from separate planes will be used to help align one another
        if not isinstance(elastic_slice,float):
            elastic_slice = 0.0
        if elastic_slice > 0.0:
            if np.abs(elastic_slice - np.sqrt(self.im_params.num_slices-3)) < 0.3:
                logging.warning("\n\nELASTIC SLICE REGULARIZATION IS SINGULAR WHEN PARAMETER IS NEAR SQRT(N_SLICES-3)"
                              f"\nYOU USED {elastic_slice}"
                              f"\nDEFAULTING TO {elastic_slice+1.0} INSTEAD\n"
                             )
                elastic_slice = elastic_slice+1.0
            simultaneous_frames = self.im_params.framelist_by_timepoint(color_channel=color_channel)
            for framelist in simultaneous_frames:
                regularized = regularize_all_tuples(
                    [reg_dict[frame] for frame in framelist],
                    self.im_params.ysize,
                    self.im_params.xsize,
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
            
            slicewise = self.im_params.framelist_by_slice()
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
        if isinstance(self.im_params.colors, list):
            # the color-1 is to 0-index
            # if color_channel is 1, offsets is [-1, 0]
            # if color_channel is 0, offsets is [0, 1]
            offsets = [(color-1) - color_channel for color in self.im_params.colors]
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