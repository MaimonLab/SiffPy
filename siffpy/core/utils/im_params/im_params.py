# Im_params object, ensures the existence of all the
# relevant data, makes a simple object to pass around
import re
import logging
from typing import Any, Union, List, Dict, Tuple, Optional
from functools import wraps

import numpy as np

from siffpy.core.utils.im_params.scanimage import ScanImageModule, ROIGroup

MULTIHARP_BASE_RES = 5 # in picoseconds WARNING BEFORE MHLIB V3 THIS VALUE IS 20. I DIDN'T THINK TO PUT THIS INFO IN THE SIFF FILE

def correct_flyback(f):
    """
    Decorator to remove flyback frames from a list of frames
    """
    @wraps(f)
    def shift_by_flybacks_wrapper(
        *args, **kwargs
        )->Union[List[int], List[List[int]]]:
        """
        Calls a function, checks that it returns the correct types
        of values (frames or lists of frames), and then removes
        flyback frames from the returned list(s) of frames.

        Otherwise, passes through
        """
        def shift_by_flyback(
                framelist : List[int],
                num_frames_per_volume : int,
                num_flyback_frames : int,
                max_frames : int,
            ):
            """
            Adds the number of flyback frames that should
            have elapsed to each frame in the list of frames
            """
            true_frames = [
                shift_frame
                for frame in framelist
                if (
                shift_frame := (
                    frame + num_flyback_frames*
                    (frame//num_frames_per_volume)
                )
                ) < max_frames
            ]
            return true_frames
        
        imparams : ImParams = args[0]
        if imparams.discard_frames:
            frames = f(*args, **kwargs)
            if all(isinstance(item, list) for item in frames):
                return [
                    shift_by_flyback(
                        framelist,
                        imparams.frames_per_volume,
                        imparams.num_discard_flyback_frames,
                        imparams.num_frames,
                    )
                    for framelist in frames
                ]
            
            true_frames = shift_by_flyback(
                frames,
                imparams.frames_per_volume,
                imparams.num_discard_flyback_frames,
                imparams.num_frames
            )
            return true_frames

        # only here if all the above is not applicable    
        return f(*args, **kwargs)
    return shift_by_flybacks_wrapper

class ImParams():
    """
    A single simple object that guarantees some core parameters
    that makes it easy to pass these things around.

    Behaves like a dict, more or less. This is partly just to
    maintain compatibility with old code when it WAS a dict,
    and partly because I think the dict-like interface is
    intuitive to people (myself included).
    """

    CHANNEL_AXIS : int = -3 # index of the color channel dimension

    def __init__(self, num_frames = None, **param_dict):
        """
        
        Initialized by reading in a param dict straight out of ScanImage,
        computes a few other useful parameters too.

        """
        self.si_modules : Dict[str, ScanImageModule]= {}

        if num_frames is not None:
            self._num_frames_from_siffio = num_frames

        if not param_dict:
            return

        try:
            self.frames_per_volume = self.num_slices * self.frames_per_slice * self.num_colors
            self.num_volumes = self.num_frames // self.frames_per_volume
        except AttributeError: # then some of the above params were not defined.
            pass
    
    def add_roi_data(self, roi_data : Dict):
        """ Adds ROI data to the ImParams object """
        self.roi_groups = {}
        for roi_group_name, roi_group_data in roi_data['RoiGroups'].items():
            if roi_group_data is not None:
                self.roi_groups[roi_group_name] = ROIGroup(roi_group_data)

    def _repr_pretty_(self, p, cycle):
        """
        Special pretty-print method for IPython notebooks.

        Not typehinted because it's IPython-specific...

        Not implemented because I don't know how to do it yet.
        """
        if cycle:
            p.text(self.__repr__())
            return
        p.text(self.__repr__())

    @property
    def discard_frames(self)->bool:
        """ Whether or not to discard frames because they are flyback frames"""
        return (
            hasattr(self, 'FastZ')
            and self.FastZ.discardFlybackFrames 
            and self.FastZ.enable
        )

    @property
    def num_discard_flyback_frames(self)->int:
        """ Number of frames per volume discarded due to flyback """
        if hasattr(self, 'FastZ'):
            return self.FastZ.numDiscardFlybackFrames*self.num_colors

    @property
    def flyback_frames(self)->List[int]:
        """ Lists the indices of flyback frames """
        if not (hasattr(self, 'FastZ') and self.FastZ.enable):
            return []
        return [
            frame for frame in range(self.num_frames)
            if (
                frame % (self.frames_per_volume + self.num_discard_flyback_frames)
                >= self.frames_per_volume
            )
        ]

    @property
    def all_frames(self)->List[int]:
        """
        Returns all _NON-FLYBACK_ frames...
        including those which will not constitute a full
        volume (e.g. the last half of a volume in the acquisition).
        """
        return list(set(range(self.num_frames)) - set(self.flyback_frames))

    @property
    def picoseconds_per_bin(self)->int:
        """ Picoseconds per photon arrival time bin """
        if hasattr(self, 'Scan2D'):
            if hasattr(self.Scan2D, 'Acq'):
                return MULTIHARP_BASE_RES*(2**(self.Scan2D.Acq.binResolution))
    
    @property
    def num_bins(self)->int:
        """ Number of photon arrival time bins """
        if hasattr(self, 'Scan2D'):
            if hasattr(self.Scan2D, 'Acq'):
                return self.Scan2D.Acq.Tau_bins

    @property
    def num_frames(self)->int:
        """ Total number of frames in the image set (INCLUDES FLYBACK)"""
        if hasattr(self, '_num_frames_from_siffio'):
            return self._num_frames_from_siffio
        if hasattr(self, 'StackManager'):
            if self.StackManager.enable:
                return (
                    self.StackManager.actualNumVolumes *
                    self.StackManager.numFramesPerVolume
                )
        return 0
    
    @property
    def frames_per_volume(self)->int:
        """ Product of elements in the volume tuple """
        fr = 1
        for x in self.volume[:-2]:
            fr *= x
        return fr

    @property
    def arrival_time_bins(self)->np.ndarray[Any, np.dtype[np.float_]]:
        """ The time bins of the arrival time histogram """
        return np.arange(self.num_bins, dtype=float)*self.picoseconds_per_bin
    
    @property
    def num_true_frames(self)->int:
        """ Number of frames that are NOT flyback """
        return int(self.num_frames * (
            self.frames_per_volume/(
                self.frames_per_volume + self.num_discard_flyback_frames
                )
            )
        )

    @property
    def num_slices(self)->int:
        """ Number of slices per volume """
        if hasattr(self, 'StackManager'):
            if self.StackManager.enable:
                if self.StackManager.stackZEndPos == self.StackManager.stackZStartPos:
                    # WARN THE USER THAT THIS IS HAPPENING
                    logging.warning(
                        "Number of slices is 1, but stack is enabled. This is likely something "
                        "set incorrectly during acquisition. Warning just in case."
                    )
                    return 1
                return self.StackManager.actualNumSlices
        return 1
    
    @property
    def frames_per_slice(self)->int:
        """ Number of frames per slice in one volume """
        if hasattr(self, 'StackManager'):
            if self.StackManager.enable:
                return self.StackManager.framesPerSlice
        return 1
    
    @property
    def step_size(self)->float:
        """ Step size between slices """
        if hasattr(self, 'StackManager'):
            if self.StackManager.enable:
                return self.StackManager.actualStackZStepSize
        return 0.0
    
    @property
    def z_vals(self)->List[float]:
        """ List of z values for each slice """
        if hasattr(self, 'StackManager'):
            if self.StackManager.enable:
                return self.StackManager.zsRelative
        return None
    
    @property
    def colors(self)->Union[List[int],int]:
        """ Can be int or list of ints, inherited from MATLAB """
        if hasattr(self, 'Channels'):
            return self.Channels.channelSave

    @property
    def color_list(self)->List[int]:
        """ Guaranteed to be a list of ints """
        if isinstance(self.colors,list):
            return self.colors
        else:
            return [self.colors]

    @property
    def zoom(self)->float:
        """ Scan zoom factor """
        if hasattr(self, 'RoiManager'):
            return self.RoiManager.scanZoomFactor
    
    @property
    def imaging_fov(self)->List[float]:
        """ Imaging field of view (in microns) -- relies on correct objective settings """
        if hasattr(self, 'RoiManager'):
            return self.RoiManager.imagingFovUm

    @property
    def xsize(self)->int:
        """ Number of pixels in the x dimension """
        if hasattr(self, 'RoiManager'):
            if self.RoiManager.mroiEnable:
                raise NotImplementedError(
                """
                These data use the mROI functionality,
                which has not yet been implemented in
                SiffPy. Let Stephen know!
                """
                )
            return self.RoiManager.pixelsPerLine

    @property
    def ysize(self)->int:
        """ Number of pixels in the y dimension """
        if hasattr(self, 'RoiManager'):
            if self.RoiManager.mroiEnable:
                raise NotImplementedError(
                """
                These data use the mROI functionality,
                which has not yet been implemented in
                SiffPy. Let Stephen know!
                """
                )
            return self.RoiManager.linesPerFrame

    @property
    def shape(self)->Tuple[int, int]:
        """ Shape of one frame: (n ysize, xsize)"""
        return (self.ysize, self.xsize)

    @property
    def volume(self)->Tuple[int, ...]:
        """
        Shape of one full volume: (num_slices, <frames_per_slice>, num_colors, ysize, xsize)
        with entries in <> only printed if they are greater than 1
        """
        ret_list = [self.num_slices]
        if self.frames_per_slice > 1:
            ret_list.append(self.frames_per_slice)
        #if self.num_colors > 1:
        ret_list.append(self.num_colors)
        ret_list += [self.ysize, self.xsize]
        return tuple(ret_list)
    
    @property
    def volume_one_color(self)->Tuple[int, ...]:
        """ Shape of one full volume in one color channel """
        if self.num_colors == 1:
            return tuple(self.volume)
        
        return tuple(self.volume[:-3] + self.volume[-2:])
    
    @property
    def single_channel_volume(self)->Tuple[int, ...]:
        """ Return the shape of one volume of one color channel (num_slices, ysize, xsize) """
        return (self.num_slices, *self.shape)
    
    @property
    def stack(self)->Tuple[int, ...]:
        """ Shape of one stack: (num_slices, frames_per_slice, ysize, xsize)"""
        ret_list = [self.num_true_frames // (self.frames_per_volume), self.num_slices]
        if self.frames_per_slice > 1 :
            ret_list += [self.frames_per_slice]
        ret_list += [self.num_colors, self.ysize, self.xsize]
        return tuple(ret_list)

    @property
    def scale(self)->List[float]:
        """
        Returns the relative scale of the spatial axes (plus a 1.0 for the time axis) in order of:
        [time , (frames_per_slice_in_units_time), (z), color, y , x]
        """
        # units of microns, except for time, which is in frames.
        ret_list = [1.0] # time axis!
        if not (self.frames_per_slice == 1):
            ret_list.append(1.0/(self.frames_per_volume)) 
        if self.num_slices > 1: # otherwise irrelevant
            step_size = self.step_size
            if self.step_size == 0: # single plane but usings stacks
                step_size = 1e-6 # arbitrary value
            ret_list.append(step_size)
        if not len(self.imaging_fov) == 4:
            raise ArithmeticError("Scale for mROI im_params not yet implemented")
        fov = self.imaging_fov
        ret_list.append(1) # color channel
        xrange = float(max([corner[0] for corner in fov]) - min([corner[0] for corner in fov]))
        yrange = float(max([corner[1] for corner in fov]) - min([corner[1] for corner in fov]))
        ret_list.append(yrange/self.ysize)
        ret_list.append(xrange/self.xsize)
        return ret_list
    @property
    def scale_force_z(self)->List[float]:
        """
        Returns the scale, but forces a z axis to be present,
        even if there is only one slice.
        """
        scale = self.scale
        if self.num_slices == 1:
            scale.insert(2, 1.0)
        return scale

    @property
    def scale_no_color_no_fps(self)->List[float]:
        """
        Returns the relative scale of the spatial axes (plus a 1.0 for the time axis) in order of:
        [time , z , y , x]
        """
        scale = self.scale
        if self.frames_per_slice != 1:
            scale.pop(1)
        # color
        scale.pop(1)
        return scale
        # ret_list = [1.0] # time axis!
        # if not (self.frames_per_slice == 1):
        #     ret_list.append(1.0/(self.frames_per_volume)) 
        # if not len(self.imaging_fov) == 4:
        #     raise ArithmeticError("Scale for mROI im_params not yet implemented")
        # if self.num_slices > 1:
        #     ret_list.append(self.step_size)
        
        # fov = self.imaging_fov
        # xrange = float(max([corner[0] for corner in fov]) - min([corner[0] for corner in fov]))
        # yrange = float(max([corner[1] for corner in fov]) - min([corner[1] for corner in fov]))
        # ret_list.append(yrange/self.ysize)
        # ret_list.append(xrange/self.xsize)
        # return ret_list

    @property
    def axis_labels(self) -> List[str]:
        """ Labels for each array axis """
        ret_list = ['Time']
        if self.frames_per_slice > 1:
            ret_list += ['Sub-slice repeats']
        if self.num_slices > 1:
            ret_list += ['Z planes']
        if self.num_colors > 1:
            ret_list += ['Color channel']
        ret_list += ['x', 'y']
        return ret_list

    @property
    def num_timepoints(self) -> int:
        """ Actually the same as num_volumes, but it's easier to read this sometimes """
        return self.num_volumes

    @property
    def num_volumes(self) -> int:
        """ Actually the same as num_timepoints, but it's easier to read this sometimes """
        return self.num_true_frames // (self.frames_per_volume)

    @property
    def lowest_color_channel(self)->int:
        """ Lowest color channel acquired (0 indexed)"""
        return min(self.color_list)

    @property
    def num_colors(self) -> int:
        """ Number of color channels acquired """
        if hasattr(self.colors, '__len__'):
            return len(self.colors)
        return 1

    @property
    def final_full_volume_frame(self)->int:
        """ Final frame that is in a full volume """
        return self.num_true_frames - self.num_true_frames % self.frames_per_volume

    ### DICT-LIKE INTERFACE ###
    ### You can treat an ImParams like a dict.
    ### This is maybe more intuitive to some people.
    def __getitem__(self, key : str) -> None:
        """ Aliases attributes to allow this to be treated like a dict """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Im param field {key} does not exist")

    def __setitem__(self, key : str, value) -> None:
        """ Aliases attribute setting so you can treat this as a dict """
        setattr(self, key.lower(), value)

    def __getattr__(self, __name: str) -> Any:
        """ Access to ScanImage module parameters like properties """
        if any(__name == module.module_name for _, module in self.si_modules.items()):
            return self.si_modules[__name]
        return super().__getattribute__(__name)

    def items(self):
        return [(attr_key, getattr(self,attr_key)) for attr_key in self.__dict__.keys()]
    
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return [getattr(self, key) for key in self.__dict__.keys()]

    def __repr__(self) -> str:
        retstr = "Image parameters: \n"
        for key in self.__dict__:
            if key == "si_modules":
                retstr += "\tScanImage modules : \n"
                for module_name, module in self.si_modules.items():
                    retstr += f"\t\t{module_name}\n"
            else:
                retstr += "\t" + str(key) + " : " + str(getattr(self,key)) + "\n" 
        return retstr
    
    @property
    def array_shape(self) -> Union[Tuple[int, int, int, int, int], Tuple[int, int, int, int, int, int]]:
        """
        Returns the shape that an array of frames would be in standard order:
        `(t, z, frames_per_slice (if greater than 1), c, y, x)`
        """
        if self.frames_per_slice > 1:
            return (
                self.num_timepoints, # t
                self.num_slices, # z
                self.frames_per_slice, # s
                self.num_colors,
                self.ysize,
                self.xsize
            )
        else:
            return (
                self.num_timepoints, # t
                self.num_slices, # z
                self.num_colors,
                self.ysize,
                self.xsize
            )

    
    # TODO: write function that gets the frames between
    # a given time start and end in experiment time (seconds)

    # def correct_flyback(self, framelist : List[int])->List[int]:
    #     """
    #     Applies local `correct_flyback` function, in case
    #     it needs to be used externally
    #     """
    #     return correct_flyback(lambda x: x)(framelist)
    
    @correct_flyback
    def flatten_by_timepoints(self,
        timepoint_start : int = 0,
        timepoint_end : Optional[int] = None,
        reference_z : Optional[int] = None,
        color_channel : Optional[int] = 0,
    )->List[int]:
        """
        Returns all frame indices within a set of timepoints.
        
        If reference_z is None, returns _all_ frames, irrespective of z.

        If color_channel (0-indexed) is None, returns all colors. But since
        timestamps for each color channel are the same, typically you expect
        NOT to use this.

        Examples
        -------
        \b

        ### A single plane with 5 z slices and 1 flyback, and 1 color channel
        ```python

        from siffpy import SiffReader
        reader : SiffReader

        reader.im_params.flatten_by_timepoints(
            timepoint_start = 0,
            timepoint_end = 10,
            reference_z = 0,
            color_channel = 0
        )
        ```
        `[0, 6, 12, 18, 24, 30, 36, 42, 48, 54]`
        
        \b

        ### All planes. Note that 5 and 11 are skipped (flyback)!
        ```python

        from siffpy import SiffReader
        reader : SiffReader

        reader.im_params.flatten_by_timepoints(
            timepoint_start = 0,
            timepoint_end = 10,
            reference_z = None,
            color_channel = 0
        )
        ```
        `[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]`
        
        """

        timestep_size = self.frames_per_volume # how many frames constitute a timepoint
        
        # figure out the start and stop points we're analyzing.
        frame_start = timepoint_start * timestep_size
        
        if timepoint_end is None:
            timepoint_end = self.num_true_frames//timestep_size
    
        if timepoint_end > self.num_true_frames//timestep_size:
            logging.warning(
                "\ntimepoint_end greater than total number of frames.\n"+
                "Using maximum number of complete timepoints in image instead.\n"
            )
            timepoint_end = self.num_true_frames//timestep_size
        
        
        frame_end = timepoint_end * timestep_size

        if reference_z is None:
            framelist = list(range(frame_start, frame_end))
        else:
            framelist = [
                frame for frame in range(frame_start, frame_end) 
                if (
                (
                    ((frame-frame_start) % timestep_size)//self.num_colors
                ) == reference_z)
            ]

        if color_channel is not None:
            framelist = framelist[color_channel::self.num_colors]
            
        return framelist

    @correct_flyback
    def framelist_by_slice(
            self,
            color_channel : Optional[int] = None,
            slice_idx : Optional[int] = None
        ) -> List[List[int]]:
        """
        List of lists, each sublist containing the frames indices that share a z slice
        
        If slice_idx is not None, returns a single list corresponding to that slice.

        Defaults to using the lowest-numbered color channel.

        color_channel is * 0-indexed *.
        """
        
        n_slices = self.num_slices
        fps = self.frames_per_slice
        n_colors = self.num_colors
        n_frames = self.num_frames

        frames_per_volume = n_slices * fps * n_colors

        n_frames -= n_frames%frames_per_volume # ensures this only goes up to full volumes.

        if (color_channel is None):
            color_channel = self.lowest_color_channel - 1 # MATLAB idx is 1-based

        frame_list = []

        all_frames : np.ndarray = np.arange( n_frames - (n_frames % frames_per_volume))
        all_frames = all_frames.reshape(n_frames//frames_per_volume, n_slices * fps, n_colors)

        if slice_idx is None:
            for slice_num in range(n_slices):
                frame_list.append(
                    all_frames[:,(slice_num*fps):((slice_num+1)*fps),color_channel].flatten().tolist()
                ) # list of lists
        else:
            if not (isinstance(slice_idx, int)):
                slice_idx = int(slice_idx)
            frame_list = all_frames[:,(slice_idx*fps):((slice_idx+1)*fps),color_channel].flatten().tolist() # just a list
        return frame_list
    
    #@correct_flyback flyback is already corrected in framelist_by_slice
    def framelist_by_slices(
            self,
            color_channel : Optional[int] = None,
            lower_bound : int = 0,
            upper_bound : Optional[int] = None,
            slices : Optional[List[int]] = None
        )->List[int]:
        """
        Flattened list of all frames corresponding to the color channel and slices provided
        """
        if upper_bound is None:
            upper_bound = self.num_timepoints
        frames = []

        if not hasattr(slices, '__len__'):
            slices = [slices]

        for slice_idx in slices:
            frames.extend(
                self.framelist_by_slice(color_channel, slice_idx)[lower_bound:upper_bound]
            )
        return frames

    @correct_flyback
    def framelist_by_color(
        self,
        color_channel : int,
        lower_bound_timepoint : int = 0,
        upper_bound_timepoint : Optional[int] = None
        )->List:
        """
        List of all frames that share a color, regardless of slice.
        Color channel is * 0-indexed * !
        """

        if color_channel >= self.num_colors:
            raise ValueError("Color channel specified larger than number of colors acquired. Color channel is indexed from 0!")

        lower_bound_frame = (color_channel +
            lower_bound_timepoint * self.frames_per_volume
        )

        if upper_bound_timepoint is None:
            return list(range(lower_bound_frame, self.num_true_frames, self.num_colors))
        else:
            upper_bound_frame = (color_channel +
                upper_bound_timepoint * self.frames_per_volume
            )
            return list(range(lower_bound_frame, upper_bound_frame, self.num_colors))

    @correct_flyback
    def framelist_by_timepoint(
        self,
        color_channel : int,
        timepoint_start : int = 0,
        timepoint_end : Optional[int] = None,
        slice_idx : Optional[int] = None)->List[List[int]]:
        """
        If slice_idx is None:
        list of lists, each containing frames that share a timepoint, i.e. same stack but different slices or colors
        
        If slice_idx is int:
        Returns all frames corresponding to a specific slice

        color_channel is * 0-indexed *.

        Examples
        ---------
        TODO
        """
        n_frames = self.num_true_frames
        n_slices = self.num_slices
        fps = self.frames_per_slice
        n_colors = self.num_colors

        frames_per_volume = n_slices * fps * n_colors

        n_frames -= n_frames%frames_per_volume # ensures this only goes up to full volumes.

        if (color_channel is None) and (n_colors == 1):
            color_channel = 0
        if (color_channel is None) and (n_colors > 1):
            color_channel = self.lowest_color_channel - 1 # MATLAB idx is 1-based

        all_frames : np.ndarray = np.arange(
            timepoint_start * frames_per_volume,
            n_frames - (n_frames % frames_per_volume)
        )
        all_frames = all_frames.reshape(n_frames//frames_per_volume, n_slices * fps, n_colors)
        if slice_idx is None:
            if timepoint_end is None:
                return all_frames[:,:,color_channel].tolist()
            else:
                return [framenum for framenum in all_frames[:,:,color_channel].tolist() if framenum < timepoint_end]
        else:
            slice_idx = int(slice_idx)
            return all_frames[:, (slice_idx*fps):((slice_idx+1)*fps), color_channel].flatten().tolist()

    @staticmethod
    def from_dict(header_dict: Dict, num_frames : Optional[int] = None)->'ImParams':
        """
        Builds an `ImParams` object from a dict of header data,
        which is easier to generate with C++ than the correct Python object
        --- though I probably will do that at some point.

        Example
        --------
        ```python
        from siffreadermodule import SiffIO
        from siffpy import ImParams
        from typing import Dict

        sio = SiffIO()
        sio.open('my_file.siff')
        header : 'Dict' = sio.get_file_header()
        im_pars : 'ImParams' = ImParams.from_dict(header)
        ```
        """
        params = ImParams(num_frames=num_frames)

        key : str
        for key, val in header_dict.items():
            split_key = key.split('.')
            if len(split_key) <= 2:
                module_name = 'base'
            else:
                module_name = split_key[1]
                if re.match(r"h[A-Z].*", module_name):
                    module_name = module_name[1:]
            if module_name not in params.si_modules:
                params.si_modules[module_name] = ScanImageModule(module_name)
            if len(split_key) == 2:
                params.si_modules[module_name].add_param(split_key[1], val)
            else:
                remaining = '.'.join(split_key[2:])
                params.si_modules[module_name].add_param(remaining, val)
        return params