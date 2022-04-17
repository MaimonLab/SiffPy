from functools import reduce
from typing import Callable, Union
import operator
from enum import Enum

import numpy as np
import holoviews as hv

from ..siffplotter import SiffPlotter, apply_opts
from ..roi_protocols.rois import ROI, subROI
from ..utils.exceptions import NoROIException
from ..utils.dims import *
from ...siffmath import fluorescence, fluorescence_fcns
from ...siffmath.fluorescence import *

__all__ = [
    'FluorescencePlotter'
]

class HeatMapDirection(Enum):
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'

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
                    'width' : 1000,
                    'colorbar' : False,
                    'colorbar_position' : 'left',
                    'colorbar_opts' : {
                        'height': 200,
                        'width' : 20,
                        'border_line_alpha':0,
                    },
                    'yaxis' : None,
                    'fontsize': 15,
                    'toolbar' : 'above',
                    'show_frame' : False,
                    'invert_yaxis' : False,
                },
                'Curve' : {
                    'line_color' : '#000000',
                    'line_width' : 1,
                    'width' : 1000,
                    'fontsize' : 15,
                    'toolbar' : 'above',
                    'show_frame' : False
                }
            }
        
    def compute_roi_timeseries(self, roi : Union[ROI, list[ROI]], *args,
        fluorescence_method : Union[str, Callable] = None,
        color_list : Union[list[int], int]= None,
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
            if not fluorescence_method in fluorescence_fcns():
                raise ValueError(
                    "Fluorescence extraction method must either be a callable or a string. Available " +
                    "string options are functions defined in siffmath.fluorescence. Those are:" +
                    reduce(operator.add, ["\n\n\t"+name for name in fluorescence_fcns()])
                )
            fluorescence_method = getattr(fluorescence,fluorescence_method)

        return fluorescence_method(roi_trace, *args, **kwargs).flatten()
        
    def compute_vector_timeseries(self,  roi : ROI = None, *args,
        fluorescence_method : Union[str,Callable] = None,
        color_list : Union[list[int],int] = None,
        **kwargs
        )-> np.ndarray:
        """
        Takes an roi ROI with subROIs and uses it to segment the data
        linked in the siffreader file into individual ROIs and
        return some analysis on each ROI. Does not store attribute
        vector_timeseries in thePlotter -- but many other functions
        that use this one do.

        Arguments
        ---------

        roi : siffpy.siffplot.roi_protocols.rois.roi.ROI

            Any ROI subclass that has a 'subROIs' attribute.

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
            raise NoROIException("No ROI provided to compute_vector_timeseries")
        
        if not hasattr(roi, 'subROIs'):
            raise NoROIException(f"Provided roi {roi} of type {type(roi)} does not have attribute 'subROIs'.")
        if not all(map(lambda x: isinstance(x, subROI), roi.subROIs)):
            raise NoROIException("Supposed subROIs (segments, columns, etc.) are not actually of type subROI. Presumed error in implementation.")

        return np.array(
            [
                self.compute_roi_timeseries(
                    sub_roi, *args,
                    fluorescence_method = fluorescence_method,
                    color_list = color_list,
                    **kwargs,
                ) for sub_roi in roi.subROIs
            ]
        )

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

        yaxis = FluorescenceAxis(yaxis_label)

        if isinstance(trace, fluorescence.FluorescenceTrace):
            yaxis_label = trace.method
            title += f", F0 = {float(trace.F0)} photons per frame"
            if trace.normalized:
                yaxis_label += "\n(normalized)"
                title += f"\nNormalization: 0 = {str(float(trace.min_val))[:5]}, 1 = {str(float(trace.max_val))[:5]}"
            yaxis.label = yaxis_label
            
        return hv.Curve(
            (
                time_axis,
                trace,
            ),
            kdims=[ImageTime()],
            vdims=[yaxis]
        ).opts(title=title)
            
    @apply_opts
    def plot_vector_timeseries(self, *args, rois : Union[ROI, list[ROI]] = None,
        direction = HeatMapDirection.HORIZONTAL, **kwargs
        )->hv.element.HeatMap:
        """
        Returns a HoloViews HeatMap object that, by default, takes ALL current ROIs that are segmented, sums
        their corresponding segments together (they must all have the same number of segments!!), and then plots
        a HeatMap object (with time in the direction of the user's choosing) of the pooled fluorescence values.

        May be passed a parameter rois to narrow the analysis to specific ROIs
        """

        if rois is None:
            rois = self.rois

        if isinstance(rois, ROI):
            if not hasattr(rois, 'subROIs'):
                raise NoROIException("Provided ROI does not have subROIs -- try segment() first!")
            rois : list[ROI] = [rois]
        for roi in rois:
            if not hasattr(roi, 'subROIs'):
                rois.remove(roi)
        
        if len(rois) == 0:
            raise NoROIException("No ROIs provided that have subROIs attribute.")

        if not(all(len(roi.subROIs) == len(rois[0].subROIs) for roi in rois)):
            raise ValueError("Not all provided ROIs have the same number of subROIs!")


        pooled  = []
        # combine like ROIs TODO: use the self-identification of each subROI!
        for segment_idx in range(len(rois[0].subROIs)):
            subROI_pool = [roi.subROIs[segment_idx] for roi in rois]
            pooled.append(self.compute_roi_timeseries(subROI_pool, *args, **kwargs))

        time_axis = self.siffreader.t_axis()

        # Now plotting stuff

        xaxis_label = "Time (sec)"

        if all(lambda x : isinstance(x, (FluorescenceTrace,FluorescenceVector)) for x in pooled):
            pooled = FluorescenceTrace(pooled)
        else:
            pooled = np.array(pooled)

        angle_axis = np.linspace(0, 2.0*np.pi, pooled.shape[0])

        if all(hasattr(subroi, 'angle') for subroi in rois[0].subROIs):
            angle_axis = np.array([subroi.angle for subroi in rois[0].subROIs])
            angle_axis = angle_axis % (2.0*np.pi)

        # TODO: COLORBAR FORMATTING

        # TODO: special treatment for FluorescenceTrace arrays
        if all(lambda x: isinstance(x, (FluorescenceTrace, FluorescenceVector)) for x in pooled):
            pass

        if (direction == HeatMapDirection.HORIZONTAL) or (direction == HeatMapDirection.HORIZONTAL.value):
            element = hv.HeatMap(
                (
                    time_axis,
                    angle_axis,
                    pooled
                ),
                kdims=[ImageTime(), AngularSpace()]
            )
        if (direction == HeatMapDirection.VERTICAL) or (direction == HeatMapDirection.VERTICAL.value):
            element = hv.HeatMap(
                (
                    angle_axis,
                    time_axis,
                    pooled
                ),
                kdims = [AngularSpace(), ImageTime()]
            ).opts(
                    {
                        'HeatMap' : {
                            'invert_yaxis' : True,
                            'height' : 1200,
                            'width' : 400,
                        }
                    }
            )
        
        return element


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
        raise NotImplementedError("Needs a rework!")
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