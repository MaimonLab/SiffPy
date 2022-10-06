import numpy as np
import holoviews as hv


from ...plotmath.fluorescence.roianalysis import *
from ...siffplotter import SiffPlotter
from ...utils import *
from ...utils.dims import *

from ....siffmath.fluorescence import FluorescenceVector
from ....siffmath.phase import estimate_phase
from ....core.utils.circle_fcns import *

__all__ = [
    'PhasePlotter'
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

    DEFAULT_OPTS = {
        'Path' : {
                    'line_color':'#3FA50F',
                    'line_width' : 1,
                    'width' : 1000,
                    'height' : 150,
                    'yaxis' : None,
                    'fontsize': 15,
                    'toolbar' : 'above',
                    'show_frame' : False,
                    'invert_yaxis' : False,
                },
    }

    INHERITED_PARAMS = [
        'local_opts',
        'siffreader',
        'reference_frames',
        'rois'
    ]

    def __init__(self, *args, fluorescence_vector : FluorescenceVector = None, **kwargs):
        f"""
        May be initialized from another SiffPlotter to inherit its
        attributes. Inherited attributes are:

            {self.__class__.INHERITED_PARAMS}
        """
        super().__init__(*args, **kwargs)
       
        if not fluorescence_vector is None:
            self.data = fluorescence_vector

    def estimate_phase(self, vector_timeseries : np.ndarray = None, phase_method : str = None, **kwargs)->np.ndarray:
        """
        Wraps the estimate_phase methods of siffmath, taking an extracted vector
        of fluorescence measurements across timepoints
        and then fitting a phase to those data. Those values are returned and stored within
        the PhasePlotter.

        Accepts vector 

        Arguments
        ---------

        vector_timeseries : np.ndarray

            Alternatively, you can skip using the ROI explicitly and just pass a
            numpy array of fluorescence data for each subROI. Default is None,
            which leads to computing its own with the compute_vector_timeseries
            function using the argument roi.

        Returns
        -------
        
        phase_estimate : np.ndarray

            A numpy array corresponding to the estimated phase of the passed
            vector_timeseries signal.
        """
        if vector_timeseries is None:
            if self.data is None:
                raise ValueError("Must provide a vector_timeseries argument if not initialized with data")
            vector_timeseries = self.data

        if phase_method is None:
            self.phase = estimate_phase(vector_timeseries, **kwargs)
        else:
            self.phase = estimate_phase(vector_timeseries, method = phase_method, **kwargs)
        return self.phase

    def visualize(self, **kwargs) -> hv.Layout:
        """
        Default behavior. kwargs are passed to estimate_phase method.

        NOT IMPLEMENTED -- HAVEN'T DECIDED WHAT ARE EVEN REASONABLE DEFAULTS.
        """
        if self.rois is None:
            raise NoROIException("No ROIs defined for this PhasePlotter.")
        
        rois = self.rois
        for roi in self.rois: 
            if not hasattr(roi, 'subROIs'):
                rois.remove(roi)
        
        if len(rois) == 0:
            raise NoROIException("No ROIs provided that have subROIs attribute.")

        if not(all(len(roi.subROIs) == len(rois[0].subROIs) for roi in rois)):
            raise ValueError("Not all provided ROIs have the same number of subROIs!")

#        pooled = [compute_vector_timeseries(self.siffreader, roi) for roi in rois]

#        pooled  = []
#        # combine like ROIs TODO: use the self-identification of each subROI!
#        for segment_idx in range(len(rois[0].subROIs)):
#            subROI_pool = [roi.subROIs[segment_idx] for roi in rois]
#            pooled.append(compute_roi_timeseries(self.siffreader, subROI_pool, *args, **kwargs))
#
#        time_axis = self.siffreader.t_axis()

        
        #phase = self.estimate_phase( **kwargs)
        #if any()
        raise NotImplementedError()

    @apply_opts
    def plot_phase(self, time : np.ndarray, phase : np.ndarray)->hv.element.Path:
        """
        Returns a Path object plotting a phase timeseries.
        """

        if not (time.shape == phase.shape):
            raise ValueError("Time and phase arguments must have same shape.")

        path_obj = hv.element.Path(
            split_angles_to_list(
                time,
                phase
            ),
            kdims = [ImageTime(), AngularSpace()]
        )

        return path_obj

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