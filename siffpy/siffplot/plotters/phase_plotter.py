from functools import reduce
from typing import Callable, Union

import numpy as np
import holoviews as hv
import operator

from ...siffplot.siffplotter import apply_opts, SiffPlotter
from ..utils import *
from ...siffplot.roi_protocols.rois import ROI, subROI
from ...siffmath import estimate_phase
from ...siffutils.circle_fcns import *

__all__ = [
    'PhasePlotter'
]

inherited_params = [
    'local_opts',
    'siffreader',
    'reference_frames',
    'rois'
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
    to inherit its properties, just as with a
    FluorescencePlotter.

    ( e.g. phase_p = PhasePlotter(siff_plotter)) )
    """

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
                raise NotImplementedError()
                #self.vector_timeseries = self.compute_vector_timeseries(roi,**kwargs)

        split_angles_to_dict()
        
        raise NotImplementedError()

    def visualize(self) -> hv.Layout:
        """ Not yet implemented """
        raise NotImplementedError()
        return super().visualize()

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