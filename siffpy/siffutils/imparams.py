# Im_params object, ensures the existence of all the
# relevant data, makes a simple object to pass around

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

        self.frames_per_volume = self.num_slices * self.frames_per_slice * n_colors

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