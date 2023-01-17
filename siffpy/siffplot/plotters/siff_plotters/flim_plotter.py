from functools import reduce
from typing import Callable, Union
import operator

import numpy as np
import holoviews as hv

from siffpy.siffplot import LATEX
from siffpy.core import FLIMParams
from siffpy.siffplot.plotters.siff_plotters.fluorescence_plotter import FluorescencePlotter
from siffpy.siffplot.roi_protocols.rois import ROI
from siffpy.siffplot.utils import apply_opts
from siffpy.siffplot.utils.exceptions import NoROIException
from siffpy.siffplot.utils.dims import *
from siffpy.siffplot.utils.enums import Direction
from siffpy.siffplot.plotmath.flim.roianalysis import *

__all__ = [
    'FlimPlotter'
]

class FlimPlotter(FluorescencePlotter):
    """
    Extends the FluorescencePlotter functionality to allow
    a focus on FLIM data relating to a 
    population of neurons or subROIs. Makes use
    of siffmath functionality, so all that can stay
    under the hood if you don't want to chase down
    the actual analyses performed.

    Can be initialized with an existing SiffPlotter
    to inherit its properties

    ( e.g. flim_p = FlimPlotter(siff_plotter)) )

    Contains methods to read frames with a SiffReader and return
    timeseries and plots. But most methods will also accept a
    previously read and processed timeseries so that you can
    customize your analysis without using the built-in methods.

    The visualize method returns a holoviews Layout object
    of a heatmap of fluorescence, customizable with various
    args and kwargs that can be read in the documentation or
    with help(flim_p.visualize).
    """

    INHERITED_PARAMS = [
        'local_opts',
        'siffreader',
        'reference_frames',
        'rois'
    ]

    def __init__(self, *args,
        FLIMParams : Union[FLIMParams,list[FLIMParams]] = None,
        **kwargs
    ):
        """
        Initialized like a FluorescencePlotter, though accepts additional
        kwarg FLIMParams
        """
        super().__init__(*args, **kwargs)
        self.FLIMParams = FLIMParams

    @apply_opts
    def plot_roi_timeseries(self, *args, rois : Union[ROI, list[ROI]] = None, **kwargs)->hv.element.Curve:
        """
        For plotting a single trace as a `hv.Curve`.

        If rois is a list, sums across all the rois first.

        Accepts args and kwargs of `FluorescencePlotter.compute_roi_timeseries`
        """
        raise NotImplementedError()
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
            if LATEX and (yaxis_label == 'dF/F'):
                yaxis_label = r"$$\Delta \text{F}/\text{F}$$"
            
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
            
    def plot_vector_timeseries(self, *args, rois : Union[ROI, list[ROI]] = None,
        direction = Direction.HORIZONTAL, 
        show_f0 : bool = False,
        **kwargs
        )->hv.element.HeatMap:
        """
        Returns a HoloViews HeatMap object that, by default, takes ALL current ROIs that are segmented, sums
        their corresponding segments together (they must all have the same number of segments!!), and then plots
        a HeatMap object (with time in the direction of the user's choosing) of the pooled fluorescence values.

        May be passed a parameter rois to narrow the analysis to specific ROIs

        Does NOT apply local_opts to the entire element, but DOES apply the local_opts to the HeatMap element.
        This allows additional annotation of the F and F0 aspects of the traces that might use different
        opts.
        """
        raise NotImplementedError()
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

        vertical : bool = (direction == Direction.VERTICAL) or (direction == Direction.VERTICAL.value)

        pooled  = []
        # combine like ROIs TODO: use the self-identification of each subROI!
        for segment_idx in range(len(rois[0].subROIs)):
            subROI_pool = [roi.subROIs[segment_idx] for roi in rois]
            pooled.append(self.compute_roi_timeseries(subROI_pool, *args, **kwargs))

        time_axis = self.siffreader.t_axis()

        # Now plotting stuff

        xaxis_label = "Time (sec)"

        if all(lambda x : isinstance(x, FluorescenceTrace) for x in pooled):
            pooled = FluorescenceTrace(pooled)
        else:
            pooled = np.array(pooled)

        angle_axis = np.linspace(0, 2.0*np.pi, pooled.shape[0])

        if all(hasattr(subroi, 'angle') for subroi in rois[0].subROIs):
            #angle_axis = np.array([subroi.angle for subroi in rois[0].subROIs])
            #angle_axis = angle_axis % (2.0*np.pi)
            pass

        # TODO: COLORBAR FORMATTING

        # TODO: special treatment for FluorescenceTrace arrays
        z_axis_name = None
        if all(lambda x: isinstance(x, FluorescenceTrace) for x in pooled):
            if not (pooled[0].method is None):
                z_axis_name = pooled[0].method
                if LATEX and (z_axis_name == 'dF/F'):
                    z_axis_name = r"$$\DeltaF/F$$"
                if pooled[0].normalized:
                    z_axis_name += "\n(normalized)"

        kdim_axes = [ImageTime(), AngularSpace()]
        
        vdim_axes = [FluorescenceAxis(z_axis_name)]

        if not vertical:
            element = hv.HeatMap(
                (
                    time_axis,
                    angle_axis,
                    pooled
                ),
                kdims= kdim_axes,
                vdims = vdim_axes,
            )
        if vertical:
            kdim_axes.reverse()
            element = hv.HeatMap(
                (
                    angle_axis,
                    time_axis,
                    pooled
                ),
                kdims = kdim_axes,
                vdims = vdim_axes,
            ).opts(
                    {
                        'HeatMap' : {
                            'invert_yaxis' : True,
                            'height' : 1200,
                            'width' : 400,
                        }
                    }
            )

        element = element.opts(self._local_opts)
        # Annotate F0        
        if show_f0 and all(isinstance(x, FluorescenceTrace) for x in pooled):
            angle_vals = element.data['Angular'].flatten()
            f0s = np.array([x.F0 for x in pooled]).flatten()
            if not vertical: 
                f0HeatMap = hv.HeatMap(
                    (
                        np.zeros_like(f0s),
                        angle_vals,
                        f0s
                    ),
                    kdims = [hv.Dimension("F0_x"), AngularSpace()],
                    vdims = [FluorescenceAxis('Photons')],
                )
            if vertical:
                f0HeatMap = hv.HeatMap(
                    (
                        angle_vals,
                        np.zeros_like(f0s),
                        f0s
                    ),
                    kdims = [AngularSpace(), hv.Dimension("F0_x")],
                    vdims = [FluorescenceAxis('Photons')],
                )
            f0HeatMap = f0HeatMap.opts(
                cmap='Greys',
                clabel = "Photons",
                clim = (0,None),
                xaxis=None,
                yaxis=None,
                colorbar=True,
                width=100,
                height=150,
                title='F0',
                colorbar_opts = {
                    'width' : 10,
                    'height' : 40,
                    'title_text_font' : 'arial',
                    'title_text_font_style' : 'normal',
                    'title_text_font_size' : '6pt',
                    'major_tick_line_width' : 0
                },
            )

            element += f0HeatMap

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
        