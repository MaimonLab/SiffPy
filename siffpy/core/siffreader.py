import copy
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

## Should I move some import statements to function definitions
from siffpy.core import io, timetools
from siffpy.core.flim import FLIMParams, FlimUnits
from siffpy.core.utils.event_stamp import EventStamp

#from siffpy.core.utils import ImParams
from siffpy.core.utils.registration_tools import RegistrationInfo, to_reg_info_class
from siffpy.core.utils.types import BoolMaskArray, ImageArray, PathLike
from siffpy.siffmath.flim import FlimTrace
from siffpy.siffmath.utils import Timeseries
from siffreadermodule import SiffIO

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
    def __init__(self, filename : Optional[PathLike] = None, open : bool = True):
        """
        Opens file `filename` if provided, otherwise creates an inactive SiffReader.
        If open is True, opens the file. If open is False, does not open the file.
        """
        self.ROI_group_data = {}
        self.opened = False
        self.debug = False
        self.events = None
        self.siffio = SiffIO()
        if isinstance(filename, Path):
            filename = str(filename)
        if filename is None:
            self.filename = ''
        elif open:
            self.open(filename)

    def __del__(self):
        """ Close file before being destroyed """
        self.close()

    def __str__(self):
        ret_string = "SIFFREADER object:\n\n"
        if self.opened:
            ret_string += "Open file: "
            ret_string += str(self.filename)
            ret_string += "\n"
        else:
            ret_string += "Inactive siffreader\n"
        #ret_string += self.im_params.__repr__()
        return ret_string

    def __repr__(self):
        """
        Prints a pretty representation of the SiffReader
        where the open file, if any, is listed first, followed by the image parameters,
        whether a registration dictionary is loaded, and whether reference images are loaded.
        
        """
        # TODO
        return self.__str__()

    def open(
        self,
        filename : Optional[PathLike] = None,
        load_time_axis : bool = False,
        ) -> None:
        """
        Opens a .siff or .tiff file with path "filename". If no value provided for filename, prompts with a file dialog.
        
        Arguments
        ------
        
        filename (optional):
            (PathLike) path to a .siff or .tiff file.

        load_time_axis (optional, bool):
            Whether or not to load the time axis of the data
            on opening. Takes longer, but then you never have to
            call it again.

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

        if load_time_axis:
            raise NotImplementedError("load_time_axis not yet implemented")
            self._time_axis_epoch = None

        #self.events = io.find_events(self.im_params, self.get_frames_metadata())
        flim_params = io.load_flim_params(filename)
        if any(flim_params):
            self._flim_params = flim_params
        #print("Finished opening and reading file")

    def close(self) -> None:
        """ Closes opened file """
        self.siffio.close()
        self.opened = False
        self.filename = ''
        if hasattr(self, 'im_params'):
            delattr(self, 'im_params')

    @property
    def flim_params(self)->Tuple[FLIMParams]:
        """ Returns the FLIMParams object if it exists """
        if hasattr(self, '_flim_params'):
            return self._flim_params
        return (None,)
    
    @flim_params.setter
    def flim_params(self, flim_params : Union[FLIMParams, str])->None:
        """ Sets the FLIMParams object """
        if isinstance(flim_params, str):
            flim_params = io.load_flim_params(flim_params)
        self._flim_params = flim_params

### TIME AXIS METHODS
    def t_axis(
        self,
        timepoint_start : int = 0,
        timepoint_end : Optional[int] = None,
        reference_z : int = 0,
        reference_time : str = 'experiment',
        ) -> np.ndarray:
        """
        Returns the time-stamps of frames. By default, returns the time stamps of all frames relative
        to acquisition start.

        Arguments
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

        Returns
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

        return self.get_time(frames = framelist, reference_time = reference_time)
            
        # return timetools.metadata_dicts_to_time(
        #     self.siffio.get_frame_metadata(frames=framelist),
        #     reference = reference_time,
        # )
    
    def save_time_axis(self)->None:
        """ Saves the time axis of all frames as nanoseconds in a numpy array """
        t_axis = self.t_axis(
            timepoint_start=0,
            timepoint_end=None,
            reference_z=0,
            reference_time='epoch'
        )
        np.save(
            str(Path(self.filename).with_suffix('_time_axis.npy')),
            t_axis,
            allow_pickle=False
        )

    @classmethod
    def load_time_axis(cls, filename : PathLike)->np.ndarray:
        """ Loads the time axis of all frames as nanoseconds from a numpy array """
        return np.load(
            str(Path(filename).with_suffix('_time_axis.npy')),
            allow_pickle=False
        )
    
    def get_time(
        self,
        frames : Optional[List[int]] = None,
        reference_time : str = "experiment"
        ) -> np.ndarray:
        """
        Gets the recorded time (in seconds) of the frame(s) numbered in list frames

        Arguments
        ------
        frames (optional, list):
            If not provided, retrieves time value of ALL frames.

        reference (optional, str):
            Referenced to start of the experiment, or referenced to epoch time.

            Possible values:
                experiment - referenced to experiment
                epoch      - referenced to epoch

        Returns
        -------------
        time (np.ndarray):
            Ordered like the list in frames (or in order from 0 to end if frames is None).
            Time into acquisition of frames (in seconds)
        """
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")
        if frames is None:
            frames = list(range(self.im_params.num_frames))

        if reference_time == 'experiment':
            return Timeseries(
                self.siffio.get_experiment_timestamps(frames = frames),
                'experiment_seconds'
            )
        
        if reference_time == 'epoch':
            # Check if mostRecentSystemTimestamp_epoch exists
            if not hasattr(self.get_frames_metadata(frames = [0])[0],'mostRecentSystemTimestamp_epoch'):
                if hasattr(self.get_frames_metadata(frames = [0])[0],'sync Stamps'):
                    warnings.warn(
                        "\n!!!!!!!!!!!\n" +
                        "No system timestamps found, so only using the laser clock. Note that this" +
                        " will drift on the scale of a few parts per million (~10 ms per hour)"
                        +"\n!!!!!!!!!!!!\n"
                    )
                    return Timeseries(
                        self.siffio.get_epoch_timestamps_laser(frames = frames),
                        'epoch_nanoseconds'
                    )
                else:
                    warnings.warn(
                        "\n!!!!!!!!!!\n" +
                        "Timestamps collected using only system clock calls : will have no _drift_"
                        + " but will contain JITTER on the order of a few milliseconds at times"
                        + "\n!!!!!!!!!!!\n"
                    )
                    return Timeseries(
                        self.siffio.get_epoch_timestamps_laser(frames = frames),
                        'epoch_nanoseconds'
                    )
            laser_time, system_time = self.siffio.get_epoch_both(frames = frames)
            # laser_time drifts with respect to the jittery system time.
            # does a linear regression to correct for this drift
            slope, _ = np.polyfit(
                laser_time,
                (
                    (laser_time-laser_time[0]).astype(float)
                    - (system_time-system_time[0]).astype(float)
                ),
                1
            )

            return Timeseries(
                laser_time - (slope*(laser_time-laser_time[0])).astype('uint64'),
                'epoch_nanoseconds'
            )
        


    @property
    def time_zero(self)->int:
        """ Returns the time zero of the experiment in epoch time """
        if hasattr(self, '_time_zero'): # only compute once, but only succeeds if we need it and it exists
            return self._time_zero
        self._time_zero = self.t_axis(0,1,0, reference_time = 'epoch')[0].astype(int)
        return self._time_zero

    @property
    def dt_volume(self)->float:
        """
        Returns the average time between volumes in seconds. Only as
        accurate as ScanImage -- does not use timestamps if the ScanImage
        RoiManager metadata is available.
        """
        if hasattr(self.im_params, 'RoiManager'):
            return 1.0/self.im_params.RoiManager.scanVolumeRate
        return np.diff(self.t_axis(reference_time = 'experiment')).mean()
        
    @property
    def dt_frame(self)->float:
        """
        Returns the average time between frames in seconds.
        Only as accurate as ScanImage -- does not use timestamps
        if the ScanImage RoiManager metadata is available.

        Might take a long time... suboptimally implemented since it
        reads the frame metadata for every frame, then parses out the
        timestamps...
        """
        if hasattr(self.im_params, 'RoiManager'):
            return self.im_params.RoiManager.scanFramePeriod
        return np.diff(self.get_time(reference = 'experiment')).mean()
    
    def sec_to_frames(self, seconds : float, base : str = 'volume')->int:
        """
        Converts a time in seconds to the number of frames that time represents.

        Can be based on the volume or individual frames.

        Arguments
        ------
        seconds : float
            Time in seconds to convert

        base : str
            Base to convert to. Can be 'volume' or 'frame'. Defaults to 'volume'.

        Returns
        -------
        frames : int
            Number of frames that time represents.

        Example
        -------
        ```python
        >>> from siffpy import SiffReader
        >>> siffreader = SiffReader('example.siff')
        >>> siffreader.dt_volume
        0.1
        >>> siffreader.sec_to_frames(1.0, base = 'volume')
        10
        ```
        """

        if base == 'volume':
            return int(seconds/self.dt_volume)
        if base == 'frame':
            return int(seconds/self.dt_frame)
        raise ValueError("Base must be 'volume' or 'frame'")

### METADATA METHODS
    def get_frames_metadata(self, frames : Optional[List[int]] = None) -> List[io.FrameMetaData]:
        if frames is None:
            frames = self.im_params.flatten_by_timepoints()
        return [io.FrameMetaData(meta_dict)
            for meta_dict in io.frame_metadata_to_dict(self.siffio.get_frame_metadata(frames=frames))
        ]

    def epoch_to_frame_time(self, epoch_time : int) -> float:
        """ Converts epoch time to frame time for this experiment (returned in seconds) """
        return timetools.epoch_to_frame_time(
            epoch_time,
            self.get_frames_metadata(frames = [0])[0]
        )
    
    def frame_time_to_epoch_time(self, frame_time : float) -> int:
        """ Converts frame time to epoch time for this experiment (returned in nanoseconds) """
        return timetools.frame_time_to_epoch(
            frame_time,
            self.get_frames_metadata(frames = [0])[0]
        )
    
    def get_appended_text(self, frames : Optional[List[int]] = None) -> List[EventStamp]:
        """
        Returns the appended text for each frame in frames.

        Arguments
        ------

        `frames : Optional[List[int]]`
            If not provided, retrieves appended text for ALL frames.

        Returns
        -------------
        `List[EventStamp]`
            List of EventStamp objects, which are dataclasses with the following fields:
                frame_number : int
                text : str
                timestamp : Optional[float] = None (in Experiment time)

        """
        if frames is None:
            frames = self.im_params.flatten_by_timepoints()
        stamp_list = [
            EventStamp(*stamp_tuple)
            for stamp_tuple in self.siffio.get_appended_text(frames = frames)
        ]

        for stamp in stamp_list:
            stamp.define_experiment_to_epoch(self.frame_time_to_epoch_time)

        return stamp_list
        

### IMAGE INTENSITY METHODS
    def get_frames(
        self,
        frames: Optional[List[int]] = None,
        registration_dict : Optional[dict] = None,
        as_array : bool = True,
        ) -> Union[List['ImageArray'], 'ImageArray']:
        
        """
        Returns the frames requested in frames keyword, or if None returns all frames.

        Wraps self.siffio.get_frames

        Arguments
        ------
        frames (optional) : List[int]
            Indices of input frames requested

        registration_dict (optional) : dict
            Registration dictionary, if used

        as_array : bool (True)
            Type of returned PyObject. Default is np.ndarray, if False will return list

        Returns
        -------------
        np.ndarray or List[np.ndarray]
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
        mask : 'BoolMaskArray',
        timepoint_start : int = 0,
        timepoint_end : Optional[int] = None,
        z_index : Optional[Union[int,List[int]]] = None,
        color_channel :  int = 1,
        registration_dict : Optional[dict] = None,
        )->'ImageArray':
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

        z_index : List[int]
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

        frames = self.im_params.framelist_by_slices(
            color_channel = color_channel-1,
            slices = z_index,
            lower_bound=timepoint_start,
            upper_bound=timepoint_end
        )
        # frames are in the wrong order for automatic masking though
        frames = np.array(frames).reshape(len(z_index),-1).T.flatten().tolist()

        summed_data = self.siffio.sum_roi(
            mask = mask,
            frames = frames,
            registration = registration_dict
        )

        # more than one slice, sum across slices
        return summed_data.reshape(
            (-1, mask.shape[0] if mask.ndim > 2 else 1)
        ).sum(axis=1)
    
    def pool_frames(self, 
        framelist : List[List[int]], 
        flim : Optional[bool] = False,
        registration : Optional[Dict] = None,
        ret_type : type = List,
        masks : Optional[List[np.ndarray]] = None 
        ) -> List[np.ndarray]:
        """
        Wraps self.siffio.pool_frames
        TODO: Docstring.
        """
            
        raise NotImplementedError("Haven't re-implemented pool_frames yet")

### FLIM METHODS
    def get_histogram(self, frames: Optional[List[int]] = None, mask : Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get just the arrival times of photons in the list frames.

        Note: uses FRAME numbers, not timepoints. So you will mix color channels
        if you're not careful.
        
        Arguments
        -----
        frames (optional, list of ints):
            Frames to get arrival times of. If NONE, collects from all frames.

        mask (optional, np.ndarray):
            z,y,x mask to apply to the data. If None, no mask is applied.

        Returns
        -------------
        histogram (np.ndarray):
            1 dimensional histogram of arrival times
        """
        if mask is not None:
            raise NotImplementedError("Masking not yet implemented in get_histogram!")
        if frames is None:
            return self.siffio.get_histogram(frames=self.im_params.all_frames)
        return self.siffio.get_histogram(frames=frames)[:self.im_params.num_bins]

    def histograms(
        self,
        color_channel : Optional['int|list'] = None,
        frame_endpoints : Sequence[Optional[int]] = (None,None)
        ) -> np.ndarray:
        """
        Returns a numpy array with arrival time histograms for all elements of the 
        keyword argument 'color_channel', which may be of type int or of type list (or any iterable).
        Each is stored on the major axis, so the returned array will be of dimensions:
        len(color_channel) x number_of_arrival_time_bins. You can define bounds in terms of numbers
        of frames (FOR THE COLOR CHANNEL, NOT TOTAL IMAGING FRAMES) with the other keyword argument
        frame_endpoints

        Arguments
        ----------
        color_channel : int or list (default None)
            0 indexed list of color channels you want returned.
            If None is provided, returns for all color channels.

        frame_endpoints : tuple(int,int) (default (None, None))
            Start and end bounds on the frames from which to collect the histograms.
        """
        # I'm sure theres a more Pythonic way... I'm ignoring it
        # Accept the tuple, and then mutate it internal
        if isinstance(frame_endpoints, tuple):
            frame_endpoints = list(frame_endpoints)
        if frame_endpoints[0] is None:
            frame_endpoints[0] = 0
        if frame_endpoints[1] is None:
            frame_endpoints[1] = int(
                self.im_params.num_volumes
                *self.im_params.frames_per_volume
                /self.im_params.num_colors
            )
        if color_channel is None:
            color_channel = self.im_params.colors if isinstance(
                self.im_params.colors, int
            ) else [c-1 for c in self.im_params.colors]
        if isinstance(color_channel, int):
            color_channel = [color_channel]

        framelists = [self.im_params.framelist_by_color(c) for c in color_channel]
        true_framelists = [
            fl[frame_endpoints[0] : frame_endpoints[1]]
            for fl in framelists
        ]
        
        return np.array([self.get_histogram(frames) for frames in true_framelists])

    def get_frames_flim(
        self,
        params : FLIMParams,
        frames: Optional[List[int]] = None,
        registration_dict : Optional[Dict] = None,
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

        registration_dict = self.registration_dict if (
                registration_dict is None
                and hasattr(self, 'registration_dict') 
            ) else registration_dict
        
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
            flim_arrays[0],
            intensity = flim_arrays[1],
            #confidence= np.array(flim_arrays[2]),
            FLIMParams = params,
            method = 'empirical lifetime',
            units = 'countbins',
        )

    def sum_mask_flim(
        self,
        params : FLIMParams,
        mask : 'BoolMaskArray',
        timepoint_start : int = 0,
        timepoint_end : Optional[int] = None,
        z_index : Optional[Union[int,List[int]]] = None,
        color_channel : int = 1,
        registration_dict : Optional[dict] = None,
        )->FlimTrace:
        """
        Computes the empirical lifetime within an ROI over timesteps.

        params determines the color channels used.

        If params is a list, returns a list of numpy arrays, each corresponding
        to the provided FLIMParams element.

        If params is a single FLIMParams object, returns a numpy array.

        Arguments
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

        color_channel : int
            Color channel to sum over. Default is 1, which means the FIRST color channel.

        registration_dict : dict
            Registration dictionary for frames.

        """

        if not isinstance(params, FLIMParams):
            raise ValueError("params argument must be a FLIMParams object")
        params = copy.deepcopy(params) 
        params.convert_units('countbins')
        timepoint_end = self.im_params.num_timepoints if timepoint_end is None else timepoint_end
        z_index = list(range(self.im_params.num_slices)) if z_index is None else z_index

        if isinstance(z_index, int):
            z_index = [z_index]

        registration_dict = self.registration_dict if registration_dict is None and hasattr(self, 'registration_dict') else registration_dict
        registration_dict = {} if registration_dict is None else registration_dict
        
        if mask.ndim != 2:
            if mask.shape[0] != self.im_params.num_slices:
                raise ValueError("Mask must have same number of z-slices as the image")

        frames = self.im_params.framelist_by_slices(
            color_channel = color_channel-1,
            slices = z_index,
            lower_bound=timepoint_start,
            upper_bound=timepoint_end
        )
        # frames are in the wrong order for automatic masking though
        frames = np.array(frames).reshape(len(z_index),-1).T.flatten().tolist()

        
        summed_intensity_data = self.siffio.sum_roi(
            mask,
            frames = frames,
            registration = registration_dict
        )

        summed_flim_data = self.siffio.sum_roi_flim(
            mask,
            params,
            frames = frames,
            registration = registration_dict
        )

        return FlimTrace(
            summed_flim_data, 
            intensity = summed_intensity_data,
            FLIMParams = params,
            method = 'empirical lifetime',
            info_string = "ROI",
            units = FlimUnits.COUNTBINS,
        ).reshape(
            (-1, mask.shape[0] if mask.ndim > 2 else 1)
        ).sum(axis=1)
                
### REGISTRATION METHODS
    def register(
        self,
        registration_method="siffpy",
        save_path : Optional[PathLike] = None, 
        alignment_color_channel : int = 0,
        **kwargs
        ) -> Dict:
        """
        Performs image registration dependent on the registration method
        called 

        Arguments
        ------
        registration_method (optional) : string
            String version of the `RegistrationInfo` class to use. Defaults to "siffpy".
        
        alignment_color_channel : int
            Color channel to use for alignment (0-indexed). Defaults to 0, the green channel, if present.

        save_path (optional) : `PathLike`
            Whether or not to save the dict. Name will be as TODO

        Other kwargs are passed to the registration method!
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
                alignment_color_channel = alignment_color_channel,
                **kwargs
            )

        # Now store the registration dict
        self.registration_info = registration_info
        
        registration_info.save(save_path = save_path)

        return self.registration_dict
    
    @property
    def registration_dict(self) -> Optional[Dict]:
        if hasattr(self, 'registration_info'):
            return self.registration_info.yx_shifts
        return None

    @property
    def reference_frames(self)->Optional[np.ndarray]:
        if hasattr(self, 'registration_info'):
            if self.registration_info.reference_frames is None:
                raise RuntimeError("No reference frames have been computed. Run register() first.")
            return self.registration_info.reference_frames
        return None
    
### IMPARAMS SHORTHAND
    @property
    def all_frames(self) -> Sequence[int]:
        return self.im_params.flatten_by_timepoints()
    
    @property
    def series_shape(self)->Sequence[int]:
        """
        Alias for (-1, *self_im_params.volume), 
        the way you should reshape most image data returned
        as flattened frames.
        """
        return (-1, *self.im_params.volume)