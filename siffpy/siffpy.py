import siffreader
import numpy as np
import tkinter as tk
import warnings

import siffutils
from siffutils.exp_math import *
from siffutils.flimparams import FLIMParams

class SiffReader(object):
    """
	Centralized Pythonic interface to the SiffReader module implemented in C.

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

    def __init__(self, flim : bool = True):
        self.file_header = {}
        self.im_params = {}
        self.opened = False
        self.filename = ''
        self.flim = flim

    def __str__(self):
        ret_string = ""
        if self.opened:
            ret_string += "Open file: "
            ret_string += self.filename
        else:
            ret_string += "Inactive siffreader"
        return ret_string
    
    def __repr__(self):
        # TODO
        return "__repr__ print statement not yet implemented"

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
            filename = tk.filedialog.askopenfilename(filetypes = (("ScanImage tiffs","*.tiff"),("ScanImage FLIM Format (siff)","*.siff")))

        if self.opened and not (filename == self.filename):
            siffreader.close()

        siffreader.open(filename)
        
        hd = siffreader.get_file_header()
        self.file_header =  {entry.split(" = ")[0] : (entry.split(" = ")[1] if (len(entry.split(" = "))>1) else None) for entry in hd["Non-varying frame data"].split("\n")}

        self.im_params = siffutils.most_important_header_data(self.file_header)
        self.im_params['NUM_FRAMES'] = siffreader.num_frames()
        self.opened = True

    def close(self) -> None:
        """ Closes opened file """
        siffreader.close()
        self.opened = False
        self.filename = ''

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
        
        timestep_size = num_slices*num_colors # how many frames constitute a timepoint
        
        # figure out the start and stop points we're analyzing.
        frame_start = timepoint_start * timestep_size
        
        if timepoint_end is None:
            frame_end = self.im_params['NUM_FRAMES']
        else:
            if timepoint_end > self.im_params['NUM_FRAMES']/timestep_size:
                warnings.warn(
                    "timepoint_end greater than total number of frames.\n"+
                    "Using maximum number of complete timepoints in image instead."
                )
                timepoint_end = self.im_params['NUM_FRAMES']/timestep_size # hope float arithmetic won't bite me in the ass here
            
            frame_end = timepoint_end * timestep_size

        # now convert to a list of all the frames whose metadata we want
        framelist = [frame for frame in range(frame_start, frame_end) 
            if (((frame-frame_start) % timestep_size) == (num_colors*reference_z))
        ]
        
        return np.array([frame['frameTimestamps_sec']
            for frame in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=framelist))
        ])
    
    def get_time(self, frames : list = None) -> np.ndarray:
        """
        Gets the recorded time (in seconds) of the frame(s) numbered in list frames

        INPUTS
        ------
        frames (optional, list):
            If not provided, retrieves time value of ALL frames.

        RETURN VALUES
        -------------
        time (np.ndarray):
            Ordered like the list in frames (or in order from 0 to end if frames is None).
            Time into acquisition of frames (in seconds)
        """
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")
        if frames is None:
            frames = list(np.range(self.im_params['NUM_FRAMES']))

        return np.array([frame['frameTimestamps_sec']
            for frame in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=frames))
        ])

    def sum_across_time(self, timespan : int = 1, 
        timepoint_start : int = 0, timepoint_end : int = None,
        z_list : list[int] = None, color_list : list[int] = None
        ) -> np.ndarray:
        """
        Sums adjacent frames in time of width "timespan" and returns a
        numpy array in standard form.

        TODO: finish this docstring.
        """

        ##### pre=processing
        if not self.opened:
            raise RuntimeError("No open .siff or .tiff")
        
        # make them iterables if they're not
        if isinstance(z_list, int):
            z_list = [z_list]
        if isinstance(color_list, int):
            color_list = [color_list]

        num_slices = self.im_params['NUM_SLICES']
        
        num_colors = 1
        if self.im_params is list:
            num_colors = len(self.im_params['COLORS'])
        
        # make them the full volume if they're None
        if z_list is None:
            z_list = list(range(num_slices))
        if color_list is None:
            color_list = list(range(num_colors))

        timestep_size = num_slices*num_colors # how many frames constitute a timepoint

        # figure out the start and stop points we're analyzing.
        frame_start = timepoint_start * timestep_size
        
        if timepoint_end is None:
            frame_end = self.im_params['NUM_FRAMES']
        else:
            if timepoint_end > self.im_params['NUM_FRAMES']/timestep_size:
                warnings.warn(
                    "timepoint_end greater than total number of frames.\n"+
                    "Using maximum number of complete timepoints in image instead."
                )
                timepoint_end = self.im_params['NUM_FRAMES']/timestep_size # hope float arithmetic won't bite me in the ass here
            
            frame_end = timepoint_end * timestep_size

        ##### the real stuff

        # now convert to a list for each set of frames we want to pool
        # list comprehension makes this... incomprehensible. So let's do it
        # the generic way.
        framelist = []
        # a list for every element of a volume
        probe_lists = [[] for idx in range(timestep_size)]

        # offsets from the frame start that we actually want, as specified by
        # z_list and color_list
        viable_indices = [z*num_colors + c for z in z_list for c in color_list]

        # step from volume to volume, recording lists of frames to pool
        for volume_start in range(frame_start,frame_end, timestep_size):
            for vol_idx in range(timestep_size):
                if vol_idx in viable_indices: # ignore undesired frames
                    probe_lists[vol_idx].append(volume_start + vol_idx)
            if (volume_start-frame_start)%timespan == 0:
                for slicelist in probe_lists:
                    if len(slicelist) > 0: # don't append ignored arrays
                        framelist.append(slicelist)
                probe_lists = [[] for idx in range(timestep_size)]
            

        # ordered by time changing slowest, then z, then color, e.g.
        # T0: z0c0, z0c1, z1c0, z1c1, ... znz0, znc1, T1: ...
        list_of_arrays = siffreader.pool_frames(framelist, flim=self.flim)

        frameshape = list_of_arrays[0].shape

        if self.flim:
            reshaped_dims = (-1, len(z_list),len(color_list),frameshape[0],frameshape[1],frameshape[2])
        else:
            reshaped_dims = (-1, len(z_list),len(color_list),frameshape[0],frameshape[1])
        
        np.array(list_of_arrays).reshape(reshaped_dims)

        return list_of_arrays

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
        pass



#########
#########
# LOCAL #
#########
#########

def suppress_warnings() -> None:
    siffreader.suppress_warnings()

def report_warnings() -> None:
    siffreader.report_warnings()

## Maybe should relocate these to siffutils.exp_math?

def channel_exp_fit(photon_arrivals : np.ndarray, num_components : int = 2, initial_fit : dict = None) -> FLIMParams:
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
    if initial_fit is None:
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

    params = FLIMParams(param_dict=initial_fit)
    params.fit_data(photon_arrivals,num_components=num_components, x0=params.param_tuple())
    return params


def fit_exp(numpy_array : np.ndarray, num_components: int = 2, fluorophores : list[str] = None) -> list[FLIMParams]:
    """
    params = siffpy.fit_exp(numpy_array, num_components=2)


    Takes a numpy array with dimensions (time, color, z, y,x,tau) or excluding any dimensions up to (y,x,tau) and
    returns a color-element list of dicts with fits of the fluorescence emission model for each color channel

    INPUTS
    ------
    numpy_array: (ndarray) An array of data formatted as any of:
        (time, color, z, y, x, tau)
        (color, z, y, x, tau)
        (z, y, x, tau)
        (y, x, tau)

    num_components: 
    
        (int) Number of exponentials in the fit

    fluorophores (list[str] or str):

        List of fluorophores, in same order as color channels. By default, is None.
        Used for initial conditions in fitting the exponentials. I doubt it's critical.
    
    RETURN VALUES
    -------------

    fits (list):
        A list of FLIMParams objects, containing fit parameters as attributes and with functionality
        to return the parameters as a tuple or dict.

    """
    # if there's a color axis, identify it first.
    color_ax = siffutils.get_color_ax(numpy_array)

    # get a tuple that is the index of all dimensions that are neither color nor tau
    non_color_non_tau_tuple = tuple([x for x in range(numpy_array.ndim-1) if (not x == color_ax)])
    
    #compress all non-color axes down to arrival time axis
    condensed = np.sum(numpy_array,axis=non_color_non_tau_tuple) # color by tau

    n_colors = 1
    if len(condensed.shape)>1:
        n_colors = condensed.shape[0]

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
    list_of_dicts_of_fluorophores = [availables[tool_name] if isinstance(tool_name,str) else None for tool_name in fluorophores]
    list_of_flimparams = [FLIMParams(param_dict = this_dict) if isinstance(this_dict, dict) else None for this_dict in list_of_dicts_of_fluorophores]
    fluorophores_tuple_list = [FlimP.param_tuple() if isinstance(FlimP, FLIMParams) else None for FlimP in list_of_flimparams]

    if n_colors>1:
        fit_list = [channel_exp_fit( condensed[x,:],num_components, fluorophores_tuple_list[x] ) for x in range(n_colors)]
    else:
        fit_list = [channel_exp_fit( condensed,num_components, fluorophores_tuple_list[0] )]

    return fit_list