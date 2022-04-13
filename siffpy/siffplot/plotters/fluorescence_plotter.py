from functools import reduce
from typing import Callable, Union
import operator, logging

import numpy as np
import holoviews as hv

from ...siffplot.siffplotter import SiffPlotter, apply_opts
from ...siffplot.roi_protocols.rois import ROI, subROI
from ...siffmath import fluorescence, fifth_percentile, string_names_of_fluorescence_fcns

__all__ = [
    'FluorescencePlotter'
]

inherited_params = [
    'local_opts',
    'siffreader',
    'reference_frames',
    'rois'
]

class FluorescencePlotter(SiffPlotter):
    """
    Extends the SiffPlotter functionality to allow
    analysis of fluorescence information relating to a 
    population of neurons or subROIs. Makes use
    of siffmath functionality, so all that can stay
    under the hood if you don't want to chase down
    the actual analyses performed.

    Can be initialized with an existing SiffPlotter
    to inherit its properties

    ( e.g. fluor_p = FluorescencePlotter(siff_plotter)) )

    Contains methods to read frames with a SiffReader and return
    timeseries and plots. But most methods will also accept a
    previously read and processed timeseries so that you can
    customize your analysis without using the built-in methods.

    The visualize method returns a holoviews Layout object
    of a heatmap of fluorescence, customizable with various
    args and kwargs that can be read in the documentation or
    with help(fluor_p.visualize).
    """

    def __init__(self, *args, **kwargs):
        f"""
        May be initialized from another SiffPlotter to inherit its
        attributes. Inherited attributes are:

            {inherited_params}
        """
        if not any([isinstance(arg,SiffPlotter) for arg in args]):
            # From scratch
            super().__init__(*args, **kwargs)
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
        
        if 'opts' in kwargs:
            self._local_opts = {**self._local_opts, **kwargs['opts']}
        else:
            self._local_opts = {**self._local_opts,
                'HeatMap' : {
                    'cmap':'Greens',
                    'width' : 1200,
                    'colorbar' : True,
                    'ylabel' : '',
                    'colorbar_position' : 'left',
                    'colorbar_opts' : {
                        'height': 200,
                        'width' : 20,
                        'border_line_alpha':0,
                    },
                    'clabel': 'Normalized dF/F',
                    'xlabel': 'Time (seconds)',
                    'fontsize': 15,
                    'toolbar' : 'above'
                },
                'Curve' : {
                    'line_color' : '#000000',
                    'width' : 1200,
                    'fontsize' : 15,
                    'toolbar' : 'above'
                }
            }
        
    def compute_roi_timeseries(self, roi : Union[ROI, list[ROI]], *args,
        fluorescence_method : Union[str, Callable] = None,
        color_list = None,
        **kwargs) -> np.ndarray:
        """
        Takes an ROI object and returns a numpy.ndarray corresponding to fluorescence method supplied applied to
        the ROI. Additional args and kwargs provided are supplied to the fluorescence method itself.

        Arguments
        ---------

        roi : siffpy.siffplot.roi_protocols.rois.roi.ROI

            Any ROI subclass or list of ROIs

        fluorescence_method : str or callable (optional)

            Which method to use to compute the timeseries from the
            frame specifications. Accepts any public function defined in 
            siffmath.fluorescence. If no argument is provided, defaults
            to dF/F with F0 defined as the fifth percentile signal in
            each ROI. NOT Normalized from 0 to 1 by default! If a callable
            is used, computes fluroescence with that function instead (with
            the expectation that this function will be the transformation
            from raw pixel photon counts into some readout a la dF/F).

        color_list : list[int], int, or None

            The color_list parameter as passed to `siffpy.SiffReader`'s sum_roi method.
            In brief, if a list 

        *args and other kwargs provided are passed directly to the fluorescence_
        method argument along with the full intensity profile, as:
            fluorescence_method(intensity, *args, **kwargs)
        .

        Returns
        -------

        roi_timeseries : np.ndarray

            Array of shape (number_of_timebins,) corresponding
            to the analysis specified on the region contained by the
            ROI provided
        """

        if isinstance(roi, list):
            if not all( isinstance(x, ROI) for x in roi ):
                raise TypeError("Not all objects provided in list are of type `ROI`.")
            roi_trace = np.sum(
                [
                    self.siffreader.sum_roi(
                        individual_roi,
                        color_list = color_list,
                        registration_dict = self.siffreader.registration_dict
                    )
                    for individual_roi in roi
                ],
                axis=0
            )
        elif isinstance(roi, ROI):
            roi_trace = self.siffreader.sum_roi(
                roi,
                color_list = color_list,
                registration_dict = self.siffreader.registration_dict
            )
        else:
            raise TypeError(f"Parameter `roi` must be of type `ROI` or a list of `ROI`s.")

        if fluorescence_method is None:
            fluorescence_method = fluorescence.dFoF         

        # Optional alternatives
        if not callable(fluorescence_method):
            if not fluorescence_method in string_names_of_fluorescence_fcns():
                raise ValueError(
                    "Fluorescence extraction method must either be a callable or a string. Available " +
                    "string options are functions defined in siffmath.fluorescence. Those are:" +
                    reduce(operator.add, ["\n\n\t"+name for name in string_names_of_fluorescence_fcns()])
                )
            fluorescence_method = getattr(fluorescence,fluorescence_method)

        return fluorescence_method(roi_trace, *args, **kwargs).flatten()
        
    def compute_vector_timeseries(self, *args, roi : ROI = None, fluorescence_method : Union[str,Callable] = None, **kwargs)-> np.ndarray:
        """
        Takes an roi ROI with subROIs and uses it to segment the data
        linked in the siffreader file into individual ROIs and
        return some analysis on each ROI. Does not store attribute
        vector_timeseries in thePlotter -- but many other functions
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
            frame specifications.
            
            Accepts a string naming any public function defined in 
            siffmath.fluorescence, even if its signature is not appropriate here.
            If no argument is provided, defaults
            to dF/F with F0 defined as the fifth percentile signal in
            each ROI.

            Any Callable provided must have a signature compliant with:
            method(intensity : np.ndarray, *args, **kwargs)

        args and other kwargs provided are passed directly to the method
        used to compute the vector_timeseries. None need to be provided
        to use the default functionality, however.

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
                raise AttributeError("No ROIs stored.")
            if not any([hasattr(individual_roi, 'subROIs') for individual_roi in self.rois]):
                raise AttributeError("No segmented subROIs in any previously stored ROIs.")
            # Takes the first segmented one if one in particular is not provided!
            for individual_roi in self.rois:
                if hasattr(individual_roi, 'subROIs'):
                    roi = individual_roi
                    break
        
        if not hasattr(roi, 'subROIs'):
            raise ValueError(f"Provided roi {roi} of type {type(roi)} does not have attribute 'subROIs'.")
        if not all(map(lambda x: isinstance(x, subROI), roi.subROIs)):
            raise ValueError("Supposed subROIs (segments, columns, etc.) are not actually of type subROI.")

        fluor = self.siffreader.get_frames() # gets all frames
        # Default behavior, warning this DOES require a well-behaved full on SiffReader
        if fluorescence_method is None:
            return fluorescence.dFoF( # computes normalized dF/F
                [fluorescence.roi_masked_fluorescence(fluor,sub) for sub in roi.subROIs], # masks every frame with the subROIs
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

        return fluorescence_method(fluor, *args, **kwargs)

    @apply_opts
    def plot_roi_timeseries(self, *args, rois : Union[ROI, list[ROI]] = None, **kwargs)->hv.element.Curve:
        """
        For plotting a single trace as a `hv.Curve`.

        If rois is a list, sums across all the rois first.

        Accepts args and kwargs of `FluorescencePlotter.compute_roi_timeseries`
        """
        if rois is None:
            if hasattr(self, 'rois'):
                rois = self.rois
        
        if not isinstance(rois, (ROI, list)):
            raise TypeError(f"Invalid rois argument. Must be of type `ROI` or a list of `ROI`s")

        trace = self.compute_roi_timeseries(
            rois,
            *args,
            **kwargs
        )

        reference_z = None
        if isinstance(rois, ROI):
            reference_z = rois.slice_idx
        if reference_z is None:
            reference_z = 0
            title = ""
        else:
            title = f"Slice number: {reference_z}"

        time_axis = self.siffreader.t_axis(reference_z = reference_z)

        # Now plotting stuff

        xaxis_label = "Time (sec)"
        yaxis_label = "Fluorescence\nmetric"

        if isinstance(trace, fluorescence.FluorescenceTrace):
            yaxis_label = trace.method
            if trace.normalized:
                yaxis_label += "\n(normalized)"
            
        return hv.Curve(
            {
                xaxis_label : time_axis,
                yaxis_label : trace,
            },
            kdims=[xaxis_label],
            vdims=[yaxis_label],
        ).opts(title=title)
            

    @apply_opts
    def visualize(self, *args, timeseries = None, t_axis = None, **kwargs)->Union[hv.Layout,hv.element.Curve]:
        """
        If timeseries has only one non-singleton dimension, returns
        a HoloViews.Curve element.

        Otherwise, returns a HoloViews Layout object that shows a heatmap visualization of the changes in fluorescence
        in the associated timeseries.

        If None is provided, then it will check if the current ROI has been segmented.
        If so, it will return a HoloViews Layout with the default analyses for vector_timeseries
        performed. If not, it will return a HoloViews Curve!

        Stores the timeseries returned, either in the vector_timseries or roi_timeseries attribute.

        Currently HoloViews only.
        """
        if timeseries is None:
            if not (hasattr(self, 'vector_timeseries') or hasattr(self,'roi_timeseries')):
                try:
                    self.vector_timeseries = self.compute_vector_timeseries(*args, **kwargs) # to store it
                    timeseries = self.vector_timeseries
                except AttributeError:
                    try:
                        self.roi_timeseries = self.compute_roi_timeseries(*args, **kwargs) # to store it
                        timeseries = self.roi_timeseries
                    except:
                        raise AttributeError(f"No adequate ROI provided to {self.__class__.__name__}")
            else:
                if hasattr(self,'vector_timeseries'):
                    timeseries = self.vector_timeseries
                else:
                    timeseries = self.roi_timeseries
            t_axis = self.siffreader.t_axis()

        if t_axis is None:
            t_axis = np.arange(0,timeseries.shape[-1])
            self.local_opts['xlabel'] = "Undefined ticks (perhaps frames?)"

        #timeseries = np.squeeze(timeseries)

        if len(timeseries.shape) == 1:
            # Return a curve
            raise NotImplementedError()
            return
        print("I finished computing the timeseries.")
        # otherwise do a heatmap
        qm = hv.HeatMap(
            (
                t_axis,
                np.linspace(0,2*np.pi,timeseries.shape[0]),
                timeseries
            )
        )

        return qm