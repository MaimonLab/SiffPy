# Im_params object, ensures the existence of all the
# relevant data, makes a simple object to pass around

import logging

import numpy as np

CORE_PARAMS = {
    'NUM_SLICES' : int,
    'FRAMES_PER_SLICE' : int,
    'Z_VALS' : list,
    'COLORS' : list
}

OPTIONAL_PARAMS = {
    'XSIZE' : int,
    'YSIZE' : int,
    'XRESOLUTION' : float,
    'YRESOLUTION' : float,
    'IMAGING_FOV' : list,
    'ZOOM' : float, 
    'PICOSECONDS_PER_BIN' : int,
    'NUM_BINS' : int,
    'NUM_FRAMES' : int,
    'STEP_SIZE' : float,
}

class ImParams():
    """
    A single simple object that guarantees some core parameters
    that makes it easy to pass these things around.

    Behaves like a dict, more or less. This is partly just to
    maintain compatibility with old code when it WAS a dict,
    and partly because I think the dict-like interface is
    intuitive to people (myself included).
    """
    def __init__(self, **param_dict):
        """
        x
        Initialized by reading in a param dict straight out of ScanImage,
        computes a few other useful parameters too.

        """

        self.num_frames : int = None # Not usually read from header directly.
        self.num_slices : int = None
        self.frames_per_slice : int = None
        self.step_size : float = None
        self.z_vals : list = None
        self.colors : list = None

        for key in CORE_PARAMS:
            if not (key in param_dict) or (key.lower() in param_dict):
                raise KeyError(f"Input param dictionary is incomplete. Lacks {key}")
            setattr(self, key.lower(), param_dict[key])
        
        for key in OPTIONAL_PARAMS:
            if (key in param_dict) or (key.lower() in param_dict):
                setattr(self, key.lower(), param_dict[key])

        try:
            self.frames_per_volume = self.num_slices * self.frames_per_slice * self.num_colors
            self.num_volumes = self.num_frames // self.frames_per_volume
        except AttributeError: # then some of the above params were not defined.
            pass

    @property
    def color_list(self)->list[int]:
        if isinstance(self.colors,list):
            return self.colors
        else:
            return [self.colors]

    @property
    def shape(self)->tuple[int]:
        return (self.ysize, self.xsize)

    @property
    def volume(self)->tuple[int]:
        ret_list = [self.num_slices]
        if self.frames_per_slice > 1:
            ret_list += [self.frames_per_slice]
        ret_list += [self.num_colors, self.ysize, self.xsize]
        return tuple(ret_list)
    
    @property
    def stack(self)->tuple[int]:
        ret_list = [self.num_frames // (self.frames_per_volume), self.num_slices]
        if self.frames_per_slice > 1 :
            ret_list += [self.frames_per_slice]
        ret_list += [self.num_colors, self.ysize, self.xsize]
        return tuple(ret_list)

    @property
    def scale(self)->list[float]:
        """
        Returns the relative scale of the spatial axes (plus a 1.0 for the time axis) in order of:
        [time , z , y , x]
        """
        # units of microns, except for time, which is in frames.
        ret_list = [1.0] # time axis!
        if not (self.frames_per_slice == 1):
            raise AttributeError("Scale attribute of im_params not implemented for more than one frame per slice.")
        if self.num_slices > 1: # otherwise irrelevant
            ret_list.append(self.step_size)
        if not len(self.imaging_fov) == 4:
            raise ArithmeticError("Scale for mROI im_params not yet implemented")
        fov = self.imaging_fov
        xrange = float(max([corner[0] for corner in fov]) - min([corner[0] for corner in fov]))
        yrange = float(max([corner[1] for corner in fov]) - min([corner[1] for corner in fov]))
        ret_list.append(yrange/self.ysize)
        ret_list.append(xrange/self.xsize)
        return ret_list

    @property
    def axis_labels(self) -> list[str]:
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
        return self.num_frames // (self.frames_per_volume)

    @property
    def lowest_color_channel(self)->int:
        return min(self.color_list)

    @property
    def num_colors(self) -> int:
        if hasattr(self.colors, '__len__'):
            return len(self.colors)
        else:
            return 1

    def __getitem__(self, key : str) -> None:
        if hasattr(self, key.lower()):
            return getattr(self, key.lower())
        else:
            raise KeyError(f"Im param field {key} does not exist")

    def __setitem__(self, key : str, value) -> None:
        setattr(self, key.lower(), value)

    def items(self):
        return [(attr_key, getattr(self,attr_key)) for attr_key in self.__dict__.keys()]
    
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return [getattr(self, key) for key in self.__dict__.keys()]

    def __repr__(self) -> str:
        retstr = "Image parameters: \n"
        for key in self.__dict__:
            retstr += "\t" + str(key) + " : " + str(getattr(self,key)) + "\n"
        return retstr
    
    def array_shape(self) -> tuple[int]:
        """ Returns the shape that an array would be in standard order """
        return (
            int(self.num_frames/(self.frames_per_volume * self.num_colors)), # t
            self.num_slices, # z
            self.num_colors,
            self.ysize,
            self.xsize
        )

    def flatten_by_timepoints(self,
        timepoint_start : int = 0,
        timepoint_end : int = None,
        reference_z : int = 0,
    )->list[int]:

        num_slices = self.num_slices
        num_colors = self.num_colors
        fps = self.frames_per_slice
        
        timestep_size = num_slices*num_colors*fps # how many frames constitute a timepoint
        
        # figure out the start and stop points we're analyzing.
        frame_start = timepoint_start * timestep_size
        
        if timepoint_end is None:
            frame_end = self.num_frames
        else:
            if timepoint_end > self.num_frames//timestep_size:
                logging.warning(
                    "\ntimepoint_end greater than total number of frames.\n"+
                    "Using maximum number of complete timepoints in image instead.\n"
                )
                timepoint_end = self.num_frames//timestep_size
            
            frame_end = timepoint_end * timestep_size

        # now convert to a list of all the frames whose metadata we want
        framelist = [frame for frame in range(frame_start, frame_end) 
            if (((frame-frame_start) % timestep_size) == (num_colors*reference_z))
        ]

        return framelist

    def framelist_by_slice(self, color_channel : int = None, upper_bound = None, slice_idx : int = None) -> list[list[int]]:
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

    def framelist_by_color(self, color_channel : int, upper_bound : int = None)->list:
        "List of all frames that share a color, regardless of slice. Color channel is * 0-indexed * !"

        if color_channel >= self.num_colors:
            raise ValueError("Color channel specified larger than number of colors acquired. Color channel is indexed from 0!")

        if upper_bound is None:
            return list(range(color_channel, self.num_frames, self.num_colors))
        else:
            return list(range(color_channel, upper_bound, self.num_colors))

    def framelist_by_timepoint(self, color_channel : int, upper_bound : int = None, slice_idx : int = None)->list[list[int]]:
        """
        If slice_idx is None:
        list of lists, each containing frames that share a timepoint, i.e. same stack but different slices or colors
        
        If slice_idx is int:
        Returns all frames corresponding to a specific slice

        color_channel is * 0-indexed *.
        """
        n_frames = self.num_frames
        n_slices = self.num_slices
        fps = self.frames_per_slice
        n_colors = self.num_colors

        frames_per_volume = n_slices * fps * n_colors

        n_frames -= n_frames%frames_per_volume # ensures this only goes up to full volumes.

        if (color_channel is None) and (n_colors == 1):
            color_channel = 0
        if (color_channel is None) and (n_colors > 1):
            color_channel = self.lowest_color_channel - 1 # MATLAB idx is 1-based

        all_frames : np.ndarray = np.arange( n_frames - (n_frames % frames_per_volume))
        all_frames = all_frames.reshape(int(n_frames/frames_per_volume), n_slices * fps, n_colors)
        if slice_idx is None:
            if upper_bound is None:
                return all_frames[:,:,color_channel].tolist()
            else:
                return [framenum for framenum in all_frames[:,:,color_channel].tolist() if framenum < upper_bound]
        else:
            if not (type(slice_idx) is int):
                slice_idx = int(slice_idx)
            return all_frames[:, (slice_idx*fps):((slice_idx+1)*fps), color_channel].flatten().tolist()



