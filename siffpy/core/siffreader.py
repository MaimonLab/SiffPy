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

        # Arguments

        * filename (optional):
            (PathLike) path to a .siff or .tiff file. Can be a string or a Path object.

        * open (optional, bool):
            Whether or not to open the file immediately. Default is True.

        # Returns

        * `SiffReader` object

        # Example

        ```python
        from siffpy import SiffReader
        # Specify file later
        reader = SiffReader()
        reader.open('example.siff')

        # Or open immediately
        reader = SiffReader('example.siff')

        # Or specify the file, but open later
        reader = SiffReader('example.siff', open = False)

        # ... do some stuff...

        reader.open()
        ```


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
        
        # Arguments
        
        * filename (optional):
            (PathLike) path to a .siff or .tiff file.

        * load_time_axis (optional, bool):
            Whether or not to load the time axis of the data
            on opening. Takes longer, but then you never have to
            call it again.

        # Example

        ```python
        from siffpy import SiffReader
        reader = SiffReader()

        reader.open('example.siff')
        ```

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

        # Arguments
        
        * timepoint_start (optional, int):
            Index of time point to start at. Note this is in units of
            TIMEPOINTS so this goes by step sizes of num_slices * num_colors!
            If no argument is given, this is treated as 0
        
        * timepoint_end (optional, int):
            Index of time point to end at. Note Note this is in units of
            TIMEPOINTS so this goes by step sizes of num_slices * num_colors!
            If no argument is given, this is the end of the file.

        * reference_z (optional, int):
            Picks the timepoint of a single slice in a z stack and only returns
            that corresponding value. Means nothing if imaging is a single plane.
            If no argument is given, assumes the first slice
        
        * reference_time (optional, str):
            Referenced to start of the experiment, or referenced to epoch time.

            Possible values:
                experiment - referenced to experiment
                epoch      - referenced to epoch

        # Returns

        * timepoints (1-d ndarray):
            Time point of the requested frames, relative to beginning of the
            image acquisition. Size will be:
                (timepoint_end - timepoint_start)*num_slices
            unless reference_z is used.

        # See also

        - `SiffReader.get_time` to get the time of individual frames

        # Examples

        ```python
        from siffpy import SiffReader
        reader = SiffReader('example.siff')

        # The data corresponds to a volume acquisition
        print( reader.im_params.array_shape )
        >> (6736, 6, 2, 256, 128)

        # Get the time of the first frame of the first 10 volumes
        print( reader.t_axis(timepoint_end = 10) )
        >> Timeseries([0.401724, 0.515611, 0.629499, 0.743387, 0.857274, 0.971162,
            1.085049, 1.198924, 1.312812, 1.426699])

        # Get the time of the 3rd frame of the first 10 volumes
        print( reader.t_axis(timepoint_end = 10, reference_z = 2) )
        >> Timeseries([0.434267, 0.548154, 0.662042, 0.77593 , 0.889817, 1.003692,
            1.11758 , 1.231467, 1.345355, 1.459242])

        # Get the time in `epoch` time of the first 10 volumes
        print( reader.t_axis(timepoint_end = 10, reference_time = 'epoch') )
        >> Timeseries([1716839309449385700, 1716839309530065982, 1716839309610746972,
            1716839309691427962, 1716839309772108244, 1716839309852789234,
            1716839309933469516, 1716839310014141297, 1716839310094822287,
            1716839310175502568], dtype=uint64)
        ```


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
        """ 
        Saves the time axis of all frames as nanoseconds in a numpy array

        # See also

        - `SiffReader.load_time_axis` to load the time axis back in.
        """
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
        """
        Loads the time axis of all frames as nanoseconds from a numpy array
        
        # See also

        - `SiffReader.save_time_axis` to save the time axis.
        """
        return np.load(
            str(Path(filename).with_suffix('_time_axis.npy')),
            allow_pickle=False
        )
    
    def get_time(
        self,
        frames : Optional[List[int]] = None,
        reference_time : str = "experiment"
        ) -> Timeseries:
        """
        Gets the recorded time (in seconds) of the frame(s) numbered in list frames

        # Arguments

        * frames (optional, list):
            If not provided, retrieves time value of ALL frames.

        * reference (optional, str):
            Referenced to start of the experiment, or referenced to epoch time.

            Possible values:
                experiment - referenced to experiment
                epoch      - referenced to epoch

        # Returns
        
        * time (`Timeseries`, an np.ndarray subclass):
            Ordered like the list in frames (or in order from 0 to end if frames is None).
            Time into acquisition of frames (in seconds) if `experiment` time, otherwise
            time into acquisition of frames (in nanoseconds) if `epoch` time.

        # See also

        - `SiffReader.t_axis` to get volume-wise timepoints without having to
        compute them yourself.

        - `SiffReader.epoch_to_frame_time` and `SiffReader.frame_time_to_epoch_time`
        to convert between epoch time and frame time.

        # Examples

        ```python
        from siffpy import SiffReader
        reader = SiffReader('example.siff')

        # Get the time of the first 10 frames
        reader.get_time(frames = list(range(10)), reference_time = 'experiment')

        >> Timeseries([0.410458, 0.410458, 0.426717, 0.426717, 0.442989, 0.442989,
            0.45926 , 0.45926 , 0.475532, 0.475532])

        reader.get_time(frames = list(range(10)), reference_time = 'epoch')

        >> Timeseries([1716838380266797800, 1716838380266797800, 1716838380266797920,
            1716838380266797920, 1716838380266798040, 1716838380266798040,
            1716838380266798160, 1716838380266798160, 1716838380266798280,
            1716838380266798280], dtype=uint64)
        ```
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
        warnings.warn(
            "Don't use `dt_frame` if there's no `RoiManager` ScanImage metadata!"
            + " It's extremely slow and badly implemented!!"
        )
        return np.diff(self.get_time(reference = 'experiment')).mean()
    
    def sec_to_frames(self, seconds : float, base : str = 'volume')->int:
        """
        Converts a time in seconds to the number of frames that time represents.

        Can be based on the volume or individual frames.

        # Arguments

        * seconds : float
            Time in seconds to convert

        * base : str
            Base to convert to. Can be 'volume' or 'frame'. Defaults to 'volume'.

        # Returns
        
        * frames : int
            Number of frames that time represents.

        # Example

        ```python
        from siffpy import SiffReader
        siffreader = SiffReader('example.siff')
        
        print(siffreader.dt_volume)
        
        >> 0.1
        
        print(siffreader.sec_to_frames(1.0, base = 'volume'))
        
        >> 10
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

        # Arguments

        * `frames : Optional[List[int]]`
            If not provided, retrieves appended text for ALL frames.

        # Returns
        
        * `List[EventStamp]`
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

        # Arguments

        * frames (optional) : `List[int]`
            Indices of input frames requested

        * registration_dict (optional) : `dict`
            Registration dictionary, if used

        * as_array : `bool` (True)
            Type of returned PyObject. Default is np.ndarray, if False will return list

        # Returns
        
        * `np.ndarray` or `List[np.ndarray]`
            Either a n_frames by y by x array or a list of numpy arrays.

        # Examples

        ```python
        from siffpy import SiffReader
        reader = SiffReader('example.siff')

        # Get the 2004th frame
        frame = reader.get_frames(frames = [2004])
        print(frame)
        
        >> array([[[1, 0, 0, ..., 0, 1, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 1],
        ...,
        [0, 0, 0, ..., 2, 0, 0],
        [0, 0, 1, ..., 2, 0, 1],
        [0, 0, 0, ..., 1, 0, 0]]], dtype=uint16)

        # Get the first 500 frames
        frames = reader.get_frames(frames = list(range(500)))

        print(frames.shape)

        >> (500, 256, 128)
        ```
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
        mask : Union['BoolMaskArray', List['BoolMaskArray']],
        timepoint_start : int = 0,
        timepoint_end : Optional[int] = None,
        z_index : Optional[Union[int,List[int]]] = None,
        color_channel :  int = 1,
        registration_dict : Optional[Dict] = None,
        framewise : bool = False,
        )->'ImageArray':
        """
        Computes the sum photon counts within a numpy mask over timesteps.
        Takes _timepoints_ as arguments, not frames. Returns a 1D array
        of summed photon counts over the entire _timepoint_ over the mask.
        If mask is 2d, applies the same mask to every frame. Otherwise,
        applies the 3d mask slices to each z slice.

        Internally calls `self._sum_mask_frames(*, **)` after computing
        the frames to sum over.

        Note: if you want to sum over multiple masks, consider
        `SiffReader.sum_masks` instead. If you pass in a `list` of
        masks, this will call `SiffReader.sum_masks` instead.

        # Arguments

        * mask : Union[np.ndarray[bool], List[np.ndarray[bool]]]
            Mask to sum over. Must be either the same shape as individual frames
            (in which case z_index is used) or have a 0th axis with length equal
            to the number of z slices. If a `list` is provided, or if the mask
            has dimension > 3, the function will sum over each mask in the list
            (presuming the slowest dimension is the mask index).

        * timepoint_start : int
            Starting timepoint for the sum. Default is 0.

        * timepoint_end : int
            Ending timepoint for the sum. Default is None, which means the last timepoint.

        * z_index : List[int]
            List of z-slices to sum over. Default is None, which means all z-slices.

        * color_channel : int
            Color channel to sum over. Default is 1, which means the FIRST color channel.
        
        * registration_dict : dict
            Registration dictionary, if there is not a stored one or if you want to use a different one.

        * framewise : bool
            If True, does not sum across timepoints.
            
        # Returns
        
        * np.ndarray
            Summed photon counts as an array of shape `(n_timepoints,)`
            (unless `framewise` is True, in which case it is `(n_frames,)`).

        # See also
        
        - `SiffReader._sum_mask_frames` to specify
        exactly which frames to sum over, rather than using
        `timepoints`.

        - `SiffReader.sum_masks` to sum over multiple masks
        in one pass of reading the file (should be far more efficient
        than iterating).

        # Examples
        
        A very simple example of summing over all the pixels:

        ```python
        from siffpy import SiffReader

        reader = SiffReader('my_file_path.siff')

        # The simplest mask of all -- 1 everywhere, on all timepoints.
        summed = reader.sum_mask(
            mask = np.ones(reader.im_params.shape).astype(bool),
        )

        print(summed.shape, reader.im_params.array_shape)
        print(summed, summed.dtype)

        >> (47375,) (47375, 1, 2, 256, 256)
        >> [13026 13157 13138 ... 13288 13419 13430] uint64
        ```

        An example of summing over only a quadrant of the image, and
        only the first 100 timepoints:
            
        ```python
        from siffpy import SiffReader

        reader = SiffReader('my_file_path.siff')

        mask = np.ones(reader.im_params.shape).astype(bool)
        mask[:mask.shape[0]//2, mask.shape[1]//2:] = False

        summed = reader.sum_mask(
            mask = mask,
            timepoint_end = 15   
        )

        print(summed.shape, summed)

        >> (15,) [9520  9845  9924  9593  9696  9977 10064 10169  9517 
        9727  9909  9827 9941  9867  9997]
        ```
        """
        if isinstance(mask, list) or (isinstance(mask, np.ndarray) and mask.ndim) > 3:
            return self.sum_masks(
                masks = mask,
                timepoint_start = timepoint_start,
                timepoint_end = timepoint_end,
                z_index = z_index,
                color_channel = color_channel,
                registration_dict = registration_dict,
                framewise = framewise
            )

        timepoint_end = self.im_params.num_timepoints if timepoint_end is None else timepoint_end
        z_index = list(range(self.im_params.num_slices)) if z_index is None else z_index

        if isinstance(z_index, int):
            z_index = [z_index]

        registration_dict = self.registration_dict if registration_dict is None and hasattr(self, 'registration_dict') else registration_dict

        if mask.ndim != 2:
            if mask.shape[0] != self.im_params.num_slices:
                raise ValueError(
                    "Mask must have same number of z-slices as the image."
                    + f" Provided masks have shape {mask.shape} and image "
                    + f"has {self.im_params.num_slices} slices."
                )

        frames = self.im_params.framelist_by_slices(
            color_channel = color_channel-1,
            slices = z_index,
            lower_bound=timepoint_start,
            upper_bound=timepoint_end
        )
        # frames are in the wrong order for automatic masking though
        frames = np.array(frames).reshape(len(z_index),-1).T.flatten().tolist()


        frames_summed = self._sum_mask_frames(
            mask = mask,
            frames = frames,
            registration_dict = registration_dict
        )
        
        if framewise:
            return frames_summed
        
        return frames_summed.reshape(
            (-1, mask.shape[0] if mask.ndim > 2 else 1)
        ).sum(axis=1)
    
    def _sum_mask_frames(
        self,
        mask : 'BoolMaskArray',
        frames : List[int],
        registration_dict : Optional[Dict] = None,
        ) -> np.ndarray:
        """
        Wraps `SiffIO.sum_roi`. Does not do any
        safety checking on the mask or frames! Use
        at your own risk! But it will pop in the stored
        `registration_dict` if it's not provided,
        I'm not that mean.

        # Arguments

        * mask : np.ndarray[bool]
            Mask to sum over. If 2d, applies the same mask to every frame.
            Otherwise, applies the 3d mask slices to each z slice sequentially
            (assumes the slowest dimension is the z index). There is no
            safety checking done to make sure that the frames cycle through
            the z slices in the order corresponding to the mask!

        * frames : List[int]
            List of frames to sum over

        * registration_dict : dict
            Registration dictionary, if there is not a stored one or
            if you want to use a custom one.
            
        # Returns

        * np.ndarray

            (`n_frames`,) framewise ROI sum

        # Examples

        ```python
        from siffpy import SiffReader
        reader = SiffReader('file_path.siff')

        # Note this is a multi-color image:
        print(reader.im_params.array_shape)

        >> (6736, 6, 2, 256, 128)

        # Here we sum over the first 100 frames in the file
        # without thinking about flyback or color channels

        naive_frames = reader._sum_mask_frames(
            mask = np.ones(reader.im_params.shape).astype(bool),
            frames = list(range(100))
        )

        # Note the empty red channel interleaved with the green channel
        print( naive_frames )
        >> array([24713,     0, 26988,     0, 25308,     0, 17528,     0, 14412,
           0, 10527,     0,    40,     0, 25611,     0, 25227,     0,
        23445,     0, 16979,     0, 13736,     0, 10608,     0,    33,
           0, 25173,     0, 24495,     0, 23705,     0, 16958,     0,
        14247,     0, 11186,     0,    50,     0, 25293,     0, 24620,
           0, 23414,     0, 16577,     0, 13895,     0, 10825,     0,
          41,     0, 25114,     0, 24773,     0, 23486,     0, 16282,
           0, 13851,     0, 10698,     0,    52,     0, 24780,     0,
        24357,     0, 22934,     0, 15772,     0, 13343,     0, 10285,
           0,    37,     0, 24456,     0, 24443,     0, 23146,     0,
        16186,     0, 13548,     0, 10575,     0,    40,     0, 24977,
           0], dtype=uint64)
        ```

        """

        registration_dict = (
            self.registration_dict
            if registration_dict is None
            and hasattr(self, 'registration_dict')
            else registration_dict
        )

        return self.siffio.sum_roi(
            mask = mask,
            frames = frames,
            registration = registration_dict
        )
    
    def sum_masks(
        self,
        masks : Union['BoolMaskArray', List['BoolMaskArray']],
        timepoint_start : int = 0,
        timepoint_end : Optional[int] = None,
        z_index : Optional[Union[int,List[int]]] = None,
        color_channel :  int = 1,
        registration_dict : Optional[dict] = None,
        framewise : bool = False,
        )->'ImageArray':
        """
        Computes the sum photon counts within a sequence of `numpy` masks over timesteps.
        The slowest dimension (i.e. `masks.shape[0]`) corresponds to iterating over
        the masks.
        Takes _timepoints_ as arguments, not frames. Returns a 2D array
        of summed photon counts over the entire _timepoint_ over each mask.
        If mask is 2d, applies the same mask to every frame. Otherwise,
        applies the 3d mask slices to each z slice.

        # Arguments

        * `masks : Union[np.ndarray[bool], List[np.ndarray[bool]]]`
            Mask to sum over. Must have `ndim > 2`.
            If it's an array, the slowest axis is
            presumed to correspond to the `mask` dimension, not the
            `z` dimension. Each individual mask can be either 2d or 3d
            but they must all be the same shape. If 3d, the slowest axis
            of each mask is presumed to iterate in `z` (or, really, in
            frames). If a `list` is provided, it is presumed that each
            element of the list is its own mask (2d or 3d).

            In other words:
            * `masks.ndim >= 3`
            * `all(mask.shape == masks[0].shape for mask in masks)`
            * `all(mask.ndim >=2 for mask in masks)`

        * `timepoint_start : int`
            Starting timepoint for the sum. Default is 0.

        * `timepoint_end : int`
            Ending timepoint for the sum. Default is None, which means the last timepoint,
            i.e. last valid full volume.

        * `z_index : List[int]`
            List of z-slices to sum over. Default is None, which means all z-slices.

        * `color_channel : int`
            Color channel to sum over. Default is 1, which means the FIRST color channel,
            (à la ScanImage).

        * `registration_dict : dict`
            Registration dictionary, if there is not a stored one or
            if you want to use a custom one.

        * `framewise : bool`
            If True, does not sum across timepoints, and returns a flat array
            by frames.
        
        # Returns

        * `np.ndarray`
            Summed photon counts as an array of shape `(n_masks, n_timepoints)`
            (or `(n_masks, n_frames)` if `framewise` is True).

        # Examples

        A simple example summing over four masks that each correspond to 
        four vertical stripes across the image in each plane:

        ```python

        from siffpy import SiffReader

        reader = SiffReader('my_file_path.siff')

        # Creates a 2d mask that is the top left quadrant of the image
        # across all z planes
        mask = np.zeros(reader.im_params.single_channel_volume).astype(bool)

        # Set left stripe to True
        mask[...,:mask.shape[-1]//4] = True

        masks = [
            np.roll(mask, shift = i*mask.shape[-1]//4, axis = -1)
            for i in range(4)
        ]

        summed = reader.sum_masks(
            masks = masks,
            timepoint_end = 5
        )

        print(summed.shape, summed)

        >> (4, 5) [[10971 13067 12082 12417 11928]
        [20523 23959 23162 21287 21274]
        [41392 42917 42119 35829 37106]
        [46590 35663 38401 45091 43896]]
        ```

        Do the same, but get the individual frames rather than the timepoints:

        ```python
        from siffpy import SiffReader

        reader = SiffReader('my_file_path.siff')

        # Creates a 2d mask that is the top left quadrant of the image
        # across all z planes
        mask = np.zeros(reader.im_params.single_channel_volume).astype(bool)

        # Set left stripe to True
        mask[...,:mask.shape[-1]//4] = True

        masks = [
            np.roll(mask, shift = i*mask.shape[-1]//4, axis = -1)
            for i in range(4)
        ]

        summed = reader.sum_masks(
            masks = masks,
            timepoint_end = 5,
            framewise = True,
        )

        print(reader.im_params.single_channel_volume)

        >> (6, 256, 128)

        print(summed.shape, summed)

        >> (4, 30) [[ 2457  2566  2381  1643  1044   880  3114  3226  2123  2072  1575   957
        3311  2720  2651  1669   986   745  3342  3483  2190  1241  1267   894
        2789  3909  1676  1557  1104   893]
        [ 2843  3921  5026  3940  2911  1882  4629  4425  5019  4149  3763  1974
        5085  4295  5381  3726  2812  1863  3539  4227  4867  3449  3241  1964
        3621  4206  4865  3622  2878  2082]
        [ 8847  8163  7758  6443  5930  4251 10788  7225  7864  7013  5858  4169
        11538  7390  8070  6171  5420  3530  6450  6830  7088  5290  5754  4417
        6845  6449  7621  5945  5521  4725]
        [10566 12338 10143  5502  4527  3514  7080 10351  8439  3745  2540  3508
        5239 10090  7603  5392  5029  5048 11962 10080  9269  6597  3633  3550
        11859 10209  9324  5158  4348  2998]]
        ```

        # See also

        - `SiffReader.sum_mask` to sum over a single mask
        """
        if isinstance(masks, list):
            masks = np.array(masks)

        timepoint_end = self.im_params.num_timepoints if timepoint_end is None else timepoint_end
        z_index = list(range(self.im_params.num_slices)) if z_index is None else z_index

        if isinstance(z_index, int):
            z_index = [z_index]

        registration_dict = self.registration_dict if registration_dict is None and hasattr(self, 'registration_dict') else registration_dict

        if masks.ndim > 3:
            if masks.shape[-3] != self.im_params.num_slices:
                raise ValueError("Mask must have same number of z-slices as the image"
                   + f" you provided masks with shape {masks.shape} with the z axis"
                   + f" presumed to be of length {masks.shape[-3]} while the number of"
                     + f" z slices in the image is {self.im_params.num_slices}"
                )

        frames = self.im_params.framelist_by_slices(
            color_channel = color_channel-1,
            slices = z_index,
            lower_bound=timepoint_start,
            upper_bound=timepoint_end
        )
        # frames are in the wrong order for automatic masking though
        frames = np.array(frames).reshape(len(z_index),-1).T.flatten().tolist()

        # Dimensions = (n_masks, n_frames)
        frames_summed = self.siffio.sum_rois(
            masks = masks,
            frames = frames,
            registration = registration_dict
        )

        if framewise:
            return frames_summed

        return frames_summed.reshape(
            (masks.shape[0], -1, masks.shape[1] if masks.ndim > 3 else 1)
        ).sum(axis=2)


        # frames_summed = self._sum_mask_frames(
        #     mask = mass,
        #     frames = frames,
        #     registration_dict = registration_dict
        # )
        
        # if framewise:
        #     return frames_summed
        
        # return frames_summed.reshape(
        #     (-1, mask.shape[0] if mask.ndim > 2 else 1)
        # ).sum(axis=1)


    
    def pool_frames(self, 
        framelist : List[List[int]], 
        flim : Optional[bool] = False,
        registration : Optional[Dict] = None,
        ret_type : type = List,
        masks : Optional[List[np.ndarray]] = None 
        ) -> List[np.ndarray]:
        """
        Wraps self.siffio.pool_frames

        NOT IMPLEMENTED
        TODO: Docstring.
        """
            
        raise NotImplementedError("Haven't re-implemented pool_frames yet")

### FLIM METHODS
    def get_histogram(self, frames: Optional[List[int]] = None, mask : Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get just the arrival times of photons in the list frames.

        Note: uses FRAME numbers, not timepoints. So you will mix color channels
        if you're not careful.
        
        # Arguments
        
        * frames (optional, list of ints):
            Frames to get arrival times of. If NONE, collects from all frames.

        * mask (optional, np.ndarray):
            z,y,x mask to apply to the data. If None, no mask is applied.

        # Returns

        * histogram (np.ndarray):
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