# Im_params object, ensures the existence of all the
# relevant data, makes a simple object to pass around

from typing import Any


CORE_PARAMS = {
    'NUM_SLICES' : int,
    'FRAMES_PER_SLICE' : int,
    'STEP_SIZE' : float,
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
    'NUM_FRAMES' : int
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
    def __init__(self, param_dict : dict):
        """
        
        Initialized by reading in a param dict straight out of ScanImage,
        computes a few other useful parameters too.

        """

        for key in CORE_PARAMS:
            if not (key in param_dict) or (key.lower() in param_dict):
                raise KeyError(f"Input param dictionary is incomplete. Lacks {key}")
            setattr(self, key.lower(), param_dict[key])
        
        for key in OPTIONAL_PARAMS:
            if (key in param_dict) or (key.lower() in param_dict):
                setattr(self, key.lower(), param_dict[key])

        try:
            n_colors = len(self.colors)
        except:
            n_colors = 1

        try:
            self.frames_per_volume = self.num_slices * self.frames_per_slice * n_colors
            self.num_volumes = self.num_frames // self.frames_per_volume
        except AttributeError: # then some of the above params were not defined.
            pass

    @property
    def shape(self):
        return (self.ysize, self.xsize)

    @property
    def volume(self):
        ret_list = [self.num_slices]
        if self.frames_per_slice > 1:
            ret_list += [self.frames_per_slice]
        ret_list += [self.num_colors, self.ysize, self.xsize]
        return tuple(ret_list)
    
    @property
    def stack(self):
        ret_list = [self.num_frames // (self.frames_per_volume), self.num_slices]
        if self.frames_per_slice > 1 :
            ret_list += [self.frames_per_slice]
        ret_list += [self.num_colors, self.ysize, self.xsize]
        return tuple(ret_list)

    @property
    def scale(self):
        # units of microns, except for time.
        ret_list = [1.0]
        if not (self.frames_per_slice == 1):
            raise AttributeError("Scale attribute of im_params not implemented for more than one frame per slice.")
        if self.num_colors > 1: # otherwise irrelevant
            ret_list.append(1.0)
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
    def axis_labels(self):
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
    def num_volumes(self):
        return self.num_frames // (self.frames_per_volume)

    @property
    def num_colors(self):
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
        n_colors = 1
        if hasattr(self.colors, '__len__'):
            n_colors = len(self.colors)
        return (
            int(self.num_frames/(self.frames_per_volume * n_colors)), # t
            self.num_slices, # z
            n_colors,
            self.ysize,
            self.xsize
        )