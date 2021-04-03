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

    def __init__(self):
        self.file_header = {}
        self.im_params = {}
        self.opened = False
        self.filename = ''

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

    def open(self, filename=None):
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

    def close(self):
        """ Closes opened file """
        siffreader.close()
        self.opened = False
        self.filename = ''

    def fit_exp(self, numpy_array, num_components=2):
        """
        params = SiffReader.fit_exp(numpy_array, num_components=2)


        Takes a numpy array with dimensions (time, color, z, y,x,tau) or excluding any dimensions up to (y,x,tau) and
        returns a color-element list of dicts with fits of the fluorescence emission model for each color channel

        INPUTS
        ------
        numpy_array: (ndarray) An array of data formatted as any of:
            (time, color, z, y, x, tau)
            (color, z, y, x, tau)
            (z, y, x, tau)
            (y, x, tau)

        num_components: (int) Number of exponentials in the fit
        
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

        # channel_exp_fit is in exp_math.py in siffutils
        if n_colors>1:
            fit_list = [channel_exp_fit( condensed[x,:],num_components ) for x in range(n_colors)]
        else:
            fit_list = [channel_exp_fit( condensed,num_components )]

        return fit_list

    def t_axis(self, timepoint_start = 0, timepoint_end = None, reference_z = 0):
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
            if ((((frame-frame_start)%num_colors) == 0) and # only first color
                (((frame-frame_start)/num_colors)%(num_slices) == reference_z) # only ref_z plane
            )
        ]
       
        return np.array([frame['frameTimestamps_sec']
            for frame in siffutils.frame_metadata_to_dict(siffreader.get_frame_metadata(frames=framelist))
        ])
        
        



    def frames_to_single_array(self, frames=None):
        """
        Retrieves the frames in the list frames and uses the information retrieved from the header
        to parse them into an appropriately shaped (i.e. "standard order" tczyxtau) single array,
        rather than returning a list of numpy arrays

        INPUTS
        ------
        frames (array-like): list or array of the frame numbers to pool. If none, returns the full file.
        """
        pass
    
    def map_to_standard_order(self, numpy_array, map_list=['time','color','z','y','x','tau']):
        """
        Takes the numpy array numpy_array and returns it in the order (time, color, z, y, x, tau).
        Input arrays of dimension < 6 will be returned as 6 dimensional arrays with singleton dimensions.

        INPUTS
        ----------
        numpy_array: (ndarray)

        map_list: (list) List of any subset of the strings:
            "time"
            "color"
            "z"
            "y"
            "x"
            "tau"
            to make it clear which indices correspond to which dimension.
            If the input array has fewer dimensions than 6, that's fine.

        RETURN VALUES
        ----------
        reshaped: (ndarray) numpy_array reordered as the standard order, (time, color, z, y, x, tau)
        """


        pass


### LOCAL FUNCTIONS


def channel_exp_fit(photon_arrivals, num_components=2):
    """
    Takes row data of arrival times and returns the param dict from an exponential fit.
    TODO: Provide more options to how fitting is done


    INPUTS
    ----------

    photon_arrivals (1-dim ndarray): Histogrammed arrival time of each photon.

    num_components (int): Number of components to the exponential TODO: enable more diversity?


    RETURN VALUES
    ----------
    FLIMParams -- (FLIMParams object)
    """
    dummy_dict = { # pretty decent guess for Camui data
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

    params = FLIMParams(param_dict=dummy_dict)
    params.fit_data(photon_arrivals,num_components=num_components)
    return params

def suppress_warnings():
    siffreader.suppress_warnings()

def report_warnings():
    siffreader.report_warnings()