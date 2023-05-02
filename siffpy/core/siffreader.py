from typing import Union
import logging
import warnings
from pathlib import Path

import numpy as np

from siffreadermodule import SiffIO

from siffpy.core import io, timetools
from siffpy.core.flim import FLIMParams, FlimUnits
from siffpy.core.utils import ImParams
from siffpy.core.utils.types import PathLike
from siffpy.core.utils.registration_tools import (
    to_reg_info_class, RegistrationInfo
)
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
    """

### INIT AND DUNDER METHODS
    def __init__(self, filename : PathLike = None):
        self.im_params : ImParams = None
        self.ROI_group_data = {}
        self.opened = False
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
        ret_string += self.im_params.__repr__()
        return ret_string

    def __repr__(self):
        """
        Prints a pretty representation of the SiffReader
        where the open file, if any, is listed first, followed by the image parameters,
        whether a registration dictionary is loaded, and whether reference images are loaded.
        
        """
        # TODO
        return self.__str__()

    def open(self,  filename : PathLike = None) -> None:
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
        filename = Path(filename)            
        if self.opened and not (filename == self.filename):
            self.siffio.close()
        
#        print(f"Opening {filename}, collecting metadata...")

        self.siffio.open(str(filename))
        self.filename = filename

        header = self.siffio.get_file_header()

        self.im_params = io.header_to_imparams(header, self.siffio.num_frames())

        self.ROI_group_data = io.header_data_to_roi_string(header)
        self.opened = True

        r_info = io.load_registration(
            self.siffio,
            self.im_params,
            filename
        )
        if r_info is not None:
            self.registration_info = r_info

        self.events = io.find_events(self.im_params, self.get_frames_metadata())

        flim_params = io.load_flim_params(filename)
        if any(flim_params):
            self.flim_params = flim_params
        #print("Finished opening and reading file")

    def close(self) -> None:
        """ Closes opened file """
        self.siffio.close()
        self.opened = False
        self.filename = ''

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
    
    def get_time(
            self,
            frames : list[int] = None,
            reference : str = "experiment"
        ) -> np.ndarray:
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
        registration_dict : dict = None,
        as_array : bool = True,
        ) -> Union[list[np.ndarray], np.ndarray]:
        """
        Returns the frames requested in frames keyword, or if None returns all frames.

        Wraps self.siffio.get_frames

        INPUTS
        ------
        frames (optional) : list[int]

            Indices of input frames requested

        registration_dict (optional) : dict

            Registration dictionary, if used

        as_array : bool (True)

            Type of returned PyObject. Default is np.ndarray, if False will return list

        RETURN VALUES
        -------------
        np.ndarray or list[np.ndarray]

            Either a n_frames by y by x array or a list of numpy arrays.
        """
        registration_dict = self.registration_dict if registration_dict is None else registration_dict
        frames = list(range(self.im_params.num_frames)) if frames is None else frames
        
        if registration_dict is None:
            framelist = self.siffio.get_frames(frames = frames, as_array = as_array)
        else:
            framelist = self.siffio.get_frames(frames = frames, registration = registration_dict, as_array = as_array)

        return framelist
    
    def sum_mask(
            self,
            mask : np.ndarray,
            timepoint_start : int = 0,
            timepoint_end : int = None,
            z_index : Union[int,list[int]] = None,
            color_channel :  int = 1,
            registration_dict : dict = None,
        )->np.ndarray:
        """
        Computes the sum photon counts within a numpy mask over timesteps.
        Takes _timepoints_ as arguments, not frames. Returns a 1D array
        of summed photon counts over the entire _timepoint_ over the mask.
        If mask is 2d, applies the same mask to every frame. Otherwise,
        applies the 3d mask slices to each z slice.

        Arguments
        ---------
        mask : np.ndarray[bool]

            Mask to sum over. Must be either the same shape as individual frames
            (in which case z_index is used) or have a 0th axis with length equal
            to the number of z slices.

        timepoint_start : int
            
            Starting timepoint for the sum. Default is 0.

        timepoint_end : int

            Ending timepoint for the sum. Default is None, which means the last timepoint.

        z_index : list[int]

            List of z-slices to sum over. Default is None, which means all z-slices.

        color_channel : int

            Color channel to sum over. Default is 1, which means the FIRST color channel.
        
        registration_dict : dict

            Registration dictionary, if there is not a stored one or if you want to use a different one.


        Returns
        -------
        np.ndarray

            Summed photon counts as an array of shape (n_timepoints, mask.shape[0])
        """

        timepoint_end = self.im_params.num_timepoints if timepoint_end is None else timepoint_end
        z_index = list(range(self.im_params.num_slices)) if z_index is None else z_index

        if isinstance(z_index, int):
            z_index = [z_index]

        registration_dict = self.registration_dict if registration_dict is None and hasattr(self, 'registration_dict') else registration_dict

        if mask.ndim != 2:
            if mask.shape[0] != self.im_params.num_slices:
                raise ValueError("Mask must have same number of z-slices as the image")

        frames = self.im_params.framelist_by_slices(color_channel = color_channel-1, slices = z_index, lower_bound=timepoint_start, upper_bound=timepoint_end)
        
        summed_data = self.siffio.sum_roi(
            mask,
            frames = frames,
            registration = registration_dict
        )

        # more than one slice, sum across slices
        return summed_data.reshape(
            (-1, mask.shape[0] if mask.ndim > 2 else 1)
        ).sum(axis=1)
   
    def pool_frames(self, 
            framelist : list[list[int]], 
            flim : bool = False,
            registration : dict = None,
            ret_type : type = list,
            masks : list[np.ndarray] = None 
        ) -> list[np.ndarray]:
        """
        Wraps self.siffio.pool_frames
        TODO: Docstring.
        """
            
        if not masks is None:
            # TODO
            raise NotImplementedError("Haven't implemented masks in pool_frames yet")
        if registration is None:
            list_of_arrays = self.siffio.pool_frames(framelist, flim=flim) 
        else:
            list_of_arrays = self.siffio.pool_frames(framelist, flim=flim, registration= registration)
        
        return list_of_arrays

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
            return self.siffio.get_histogram(frames=self.im_params.all_frames)
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
        to the frames requested. Units of the FlimTrace
        are 'countbins'.
        """
     
        if frames is None:
            frames = list(range(self.im_params.num_frames))

        registration_dict = self.registration_dict if registration_dict is None and hasattr(self, 'registration_dict') else registration_dict
        if registration_dict is None:
            flim_arrays = self.siffio.flim_map(
                params,
                frames = frames, 
                confidence_metric=confidence_metric
            )
        else:
            flim_arrays = self.siffio.flim_map(
                params,
                frames = frames,
                registration = registration_dict,
                confidence_metric=confidence_metric
            )

        return FlimTrace(
            np.array(flim_arrays[0]),
            intensity = np.array(flim_arrays[1]),
            #confidence= np.array(flim_arrays[2]),
            FLIMParams = params,
            method = 'empirical lifetime',
            units = 'countbins',
        )

    def sum_mask_flim(self,
            params : 'FLIMParams',
            mask : np.ndarray[bool],
            timepoint_start : int = 0,
            timepoint_end : int = None,
            color_channel : int = 0,
            registration_dict : dict = None,
        )->'FlimTrace':
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

        mask : np.ndarray[bool]

            A mask that defines the boundaries of the region being considered.
            May be 2d (in which case it is applied to all z-planes) or higher
            dimensional (in which case it must have the same number of z-planes
            as the image data frames requested).

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

        if not isinstance(params, FLIMParams):
            raise ValueError("params argument must be a FLIMParams object")
        
        if not (params.units.value == 'countbins'):
            params.convert_units('countbins')

        if timepoint_end is None:
            timepoint_end = self.im_params.num_timepoints

        if registration_dict is None:
            if hasattr(self,'registration_dict'):
                registration_dict = self.registration_dict

        frames = self.im_params.framelist_by_slice(color_channel = color_channel)
        frames = [individual_frame for slicewise in frames for individual_frame in slicewise[timepoint_start:timepoint_end]]

        summed_intensity_data = self.siffio.sum_roi(
            mask,
            frames = frames,
            registration = registration_dict
        ).reshape(
            (-1, self.im_params.num_slices if mask.ndim == 2 else mask.shape[0])
        )

        summed_flim_data = self.siffio.sum_roi_flim(
            mask,
            params,
            frames = frames,
            registration = registration_dict
        ).reshape(
            (-1, self.im_params.num_slices if mask.ndim == 2 else mask.shape[0])
        )

        return FlimTrace(
            summed_flim_data, 
            intensity = summed_intensity_data,
            FLIMParams = params,
            method = 'empirical lifetime',
            info_string = "ROI",
            units = FlimUnits.COUNTBINS,
        ).sum(axis=1)
                
### REGISTRATION METHODS
    def register(
        self,
        registration_method="siffpy",
        color_channel : int = 0,
        save_path : PathLike = None, 
        align_z : bool = False,
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
                'siffpy' -- A minimal version of suite2p's alignment procedure

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
        """

        logging.warn("\n\n \t Don't forget to fix the zplane alignment!!")
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")

        registration_info : RegistrationInfo = to_reg_info_class(
            registration_method
        )(
            self.siffio,
            self.im_params
        )

        # Gets rid of unhelpful scipy warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            registration_info.register(
                self.siffio,
                alignment_color_channel=color_channel,
                align_z = align_z,
                **kwargs
            )

        # Now store the registration dict
        self.registration_info = registration_info
        
        registration_info.save(save_path = save_path)

        return self.registration_dict
    
    @property
    def registration_dict(self) -> dict:
        if hasattr(self, 'registration_info'):
            return self.registration_info.yx_shifts
        return None

    @property
    def reference_frames(self)->np.ndarray:
        if hasattr(self, 'registration_info'):
            if self.registration_info.reference_frames is None:
                raise RuntimeError("No reference frames have been computed. Run register() first.")
            return self.registration_info.reference_frames
        return None