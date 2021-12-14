from functools import reduce
from siffpy.siffmath import fluorescence
from typing import Callable, Union
import numpy as np
import holoviews as hv
import inspect, operator

from ...siffplot.siffplotter import SiffPlotter, apply_opts
from ...siffplot.roi_protocols.rois import ROI, subROI
from ...siffmath import estimate_phase, fluorescence

__all__ = [
    'PhasePlotter'
]

inherited_params = [
    'local_opts',
    'siffreader',
    'reference_frames',
    'rois',
    'viewers',
    'use_napari'
]

class PhasePlotter(SiffPlotter):
    """
    Extends the SiffPlotter functionality to allow
    analysis of phase information relating to a 
    population of neurons or subROIs. Makes use
    of siffmath functionality, so all that can stay
    under the hood if you don't want to chase down
    the actual analyses performed.

    Can be initialized with an existing SiffPlotter
    to inherit its properties

    ( e.g. phase_p = PhasePlotter(siff_plotter)) )
    """

    def __init__(self, *args, **kwargs):
        if not any([isinstance(arg,SiffPlotter) for arg in args]):
            # From scratch
            super().__init__(self, *args, **kwargs)
        else:
            for arg in args:
                # Iterate until you get to the first SiffPlotter object.
                if isinstance(arg, SiffPlotter):
                    plotter = arg
                    break
            
            # inherits parameters from the provided plotter
            for param in inherited_params:
                if hasattr(plotter, param):
                    setattr(self, param, getattr(plotter, param))
        
    def compute_vector_timeseries(self, *args, roi : ROI = None, fluorescence_method : Union[str,Callable] = None, **kwargs)-> np.ndarray:
        """
        Takes an roi ROI with subROIs and uses it to segment the data
        linked in the siffreader file into individual ROIs and
        return some analysis on each ROI. Does not store attribute
        vector_timeseries in PhasePlotter -- but many other functions
        that use this one do.

        Arguments
        ---------

        roi : siffpy.siffplot.roi_protocols.rois.roi.ROI (optional)

            Any ROI subclass that has a 'subROIs' attribute. If none
            is provided, will look at this SiffPlotter's rois attribute
            to see if any meet the necessary criteria, and uses the first
            one of those that it finds.

        fluorescence_method : str or callable (optional)

            Which method to use to compute the vector_timeseries from the
            frame specifications. Accepts any public function defined in 
            siffmath.fluorescence. If no argument is provided, defaults
            to dF/F with F0 defined as the fifth percentile signal in
            each ROI.

        *args and other kwargs provided are passed directly to the method
        used to compute the vector_timeseries

        Returns
        -------

        vector_timeseries : np.ndarray

            Array of shape (number_of_subROIs, number_of_timebins) corresponding
            to the analysis specified on each of the subROIs of the argument
            ROI provided.
        """
        if roi is None:
            # See if any of the already stored ROIs work.
            if self.rois is None:
                raise AttributeError("No ROIs stored in PhasePlotter.")
            if not any([hasattr(individual_roi, 'subROIs') for individual_roi in self.rois]):
                raise AttributeError("No segmented subROIs in any previously stored ROIs.")
            # Takes the first segmented one if one in particular is not provided!
            for individual_roi in self.rois:
                if hasattr(individual_roi, 'subROIs'):
                    roi = individual_roi
                    break
        
        if not hasattr(roi, 'subROIs'):
            raise ValueError(f"Provided roi {roi} of type {type(roi)} does not have attribute 'subROIs'.")
        
        if not all(isinstance(roi.subROIs, subROI)):
            raise ValueError("Supposed subROIs (segments, columns, etc.) are not actually of type subROI.")

        # Default behavior
        if fluorescence_method is None:
            fluor = self.siffreader.get_frames() # gets all frames
            return fluorescence.dFoF( # computes normalized dF/F
                fluorescence.roi_masked_fluorescence(fluor,roi.subROIs), # masks every frame with the subROIs
                normalized=True, # Normalized from 0-ish to 1-ish.
                Fo = fifth_percentile # defined locally
            )

        # Optional alternatives
        if not callable(fluorescence_method):
            if not fluorescence_method in string_names_of_fluorescence_fcns():
                raise ValueError(
                    "Fluorescence extraction method must either be a callable or a string. Available " +
                    "string options are functions defined in siffmath.fluorescence. Those are:" +
                    reduce(operator.add, ["\n\n\t"+name for name in string_names_of_fluorescence_fcns()])
                )
            fluorescence_method = getattr(fluorescence,fluorescence_method)

        return fluorescence_method(*args, **kwargs)

    def estimate_phase(self, roi : ROI = None, vector_timeseries : np.ndarray = None, phase_method : str = None, **kwargs)->np.ndarray:
        """
        Wraps the estimate_phase methods of siffmath, taking a segmented
        ROI, using it to extract fluorescence data from the siffreader frames,
        and then fitting a phase to those data (with the option of returning
        error estimates too). Those values are returned and stored within
        the PhasePlotter.

        Arguments
        ---------

        roi : siffpy.siffplot.roi_protocols.rois.roi.ROI (optional)

            Any ROI subclass that has a 'subROIs' attribute. If None
            is provided, will look at this SiffPlotter's rois attribute
            to see if any meet the necessary criteria, and uses the first
            one of those that it finds. Default is None.

        vector_timeseries : np.ndarray (optional)

            Alternatively, you can skip using the ROI explicitly and just pass a
            numpy array of fluorescence data for each subROI. Default is None,
            which leads to computing its own with the compute_vector_timeseries
            function using the argument roi.

        Returns
        -------
        NotImplementedError
        """
        if vector_timeseries is None:
            if not hasattr(self, 'vector_timeseries'):
                self.vector_timeseries = self.compute_vector_timeseries(roi,**kwargs)


        
        raise NotImplementedError()

    @apply_opts
    def wrapped_error_plot(x_axis : np.ndarray, central_data : np.ndarray, error : np.ndarray,
        bounds = (0, 2*np.pi), **kwargs)->hv.Overlay:
        """
        Returns a Holoviews Overlay of three Holoviews Spreads: one that is standard,
        one that wraps the error from the bottom bound up to the top, and one that
        wraps the error from the top bound down to the bottom.

        Arguments
        ---------

        x_axis : np.ndarray

            The x axis of the plot. This should be a 1d np.ndarray

        central_data : np.ndarray

            The main data around which the spread is to be drawn. Don't worry, the
            line color is set to None by default. This should be one-dimensional.

        error : np.ndarray

            The size of the error bars. This can either be 1d or 2d, but I will require
            that it have the 0th index correspond to the lower error and the 1th
            correspond to the upper error.

        bounds : tuple (optional)

            The lower and upper bound around which to wrap. Defaults to (0, 2*np.pi)

        Accepts all kwargs of hv.Spread

        Returns
        -------

        wrapped_error : hv.Overlay

            An Overlay object just showing the error bars, wrapped around the
            bounds on the y axis.
        """
        # I know I'm always going to forget to format the error so I'm just going
        # to make it work for lots of formats
        if len(error.shape) > 2:
            raise ValueError("Array for error bars is not 1 or 2 dimensional.")
        elif len(error.shape) == 1:
            upper_error = error/2.0
            lower_error = error/2.0
        elif min(error.shape) == 1:
            upper_error = error.flatten()/2.0
            lower_error = error.flatten()/2.0
        else:
            data_idx = np.where(np.array(error.shape) == central_data.shape[0])[0][0]
            if data_idx == 0:
                lower_error = error[:, 0]
                upper_error = error[:, 1]
            if data_idx == 1:
                lower_error = error[0,:]
                upper_error = error[1,:]

        overflow_idx = np.where(central_data + upper_error > bounds[1])
        underflow_idx = np.where(central_data - lower_error < bounds[0])

        overflow_array = bounds[0]*np.ones_like(central_data)
        overflow_err = np.zeros_like(central_data)
        overflow_err[overflow_idx] = (central_data+upper_error)[overflow_idx] - bounds[1] # how much it exceeds the bound

        underflow_array = bounds[1]*np.ones_like(central_data)
        underflow_err = np.zeros_like(central_data)
        underflow_err[underflow_idx] = bounds[0] - (central_data-lower_error)[underflow_idx] # How much lower it is than the bound

        wrapped_error = hv.Spread((x_axis, central_data, lower_error, upper_error), vdims=['y', 'yerrneg', 'yerrpos'], **kwargs).opts(line_color=None)
        wrapped_error *= hv.Spread((x_axis,overflow_array,overflow_err) **kwargs).opts(line_color=None)
        wrapped_error *= hv.Spread((x_axis,underflow_array,underflow_err) **kwargs).opts(line_color=None)

        wrapped_error = wrapped_error.opts(ylim=bounds)
        return wrapped_error
        
### LOCAL

def string_names_of_fluorescence_fcns(print_docstrings : bool = False) -> list[str]:
    """
    List of public functions available from fluorescence
    submodule. Seems a little silly since I can just use
    the __all__ but this way I can also print the
    docstrings.
    """
    fcns = inspect.getmembers(fluorescence, inspect.isfunction)
    if print_docstrings:
        return ["\033[1m" + fcn[0] + ":\n\n\t" + str(inspect.getdoc(fcns[1])) + "\033[0m" for fcn in fcns if fcn[0] in fluorescence.__dict__['__all__']] 
    return [fcn[0] for fcn in fcns if fcn[0] in fluorescence.__dict__['__all__']]


def fifth_percentile(rois : np.ndarray) -> np.ndarray:
    sorted_array = np.sort(rois,axis=1)
    return sorted_array[:, rois.shape[0]//20]