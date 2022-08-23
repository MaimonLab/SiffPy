from typing import  Union

import numpy as np
import holoviews as hv

from ....siffplot import LATEX
from ...siffplotter import SiffPlotter, apply_opts
from ...plotmath.fluorescence.roianalysis import *
from ...roi_protocols.rois import ROI
from ...utils.exceptions import NoROIException
from ...utils.dims import *
from ...utils.enums import Direction
from ....siffmath.fluorescence import *

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

    DEFAULT_OPTS = {
        'HeatMap' : {
                    'cmap':'Greens',
                    'width' : 1000,
                    'height' : 150,
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
                    #'hooks' : [bounds_hook],
            },
        'Curve' : {
            'line_color' : '#000000',
            'line_width' : 1,
            'width' : 1000,
            'fontsize' : 15,
            'toolbar' : 'above',
            'show_frame' : False,
            'height' : 200,
            #'hooks' : [bounds_hook, ],
            },
    }

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
                **self.__class__.DEFAULT_OPTS
            }
        
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

        trace = compute_roi_timeseries(
            self.siffreader,
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

        if isinstance(trace, FluorescenceTrace):
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
            pooled.append(compute_roi_timeseries(self.siffreader, subROI_pool, *args, **kwargs))

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
            f0HeatMap = self.plot_f0(
                pooled,
                angular_axis = angle_axis,
                vertical = not vertical, # inverse direction
                opts = {
                    'cmap' : "Greys",
                    'clabel' : "Photons",
                    'clim' : (0, None),
                    'xaxis' : None,
                    'yaxis' : None,
                    'colorbar' : True,
                    'width' : 100,
                    'height' : 150,
                    'title' : 'F0',
                    'colorbar_opts' : {
                        'width' : 10,
                        'height' : 40,
                        'title_text_font' : 'arial',
                        'title_text_font_style' : 'normal',
                        'title_text_font_size' : '6pt',
                        'major_tick_line_width' : 0,
                    },
                },
            )
            element += f0HeatMap

        return element

    @classmethod
    def plot_f0(self,
            f_vec : Union[FluorescenceVector,list[FluorescenceTrace]],
            angular_axis : tuple = None,
            vertical : bool = True, 
            opts : dict = None
        )->hv.Layout:
        """
        Class method because it doesn't rely on any internal functionality.

        Takes an iterable of FluorescenceTraces and returns a heatmap showing
        all of their F0 values.

        TODO: FINISH DOCSTRING
        """
        if not (
            isinstance(f_vec, FluorescenceVector)
            or
            all(isinstance(x, FluorescenceTrace) for x in f_vec)
        ):
            raise ValueError("plot_f0 only defined for "
                "FluorescenceVector argument or"
                "iterable of FluorescenceTrace"
            )

        f0s = np.array([x.F0 for x in f_vec]).flatten()
        if angular_axis is None:
            angular_axis = [x.angle for x in f_vec]
        if vertical: 
            f0HeatMap = hv.HeatMap(
                (
                    np.zeros_like(f0s),
                    angular_axis,
                    f0s
                ),
                kdims = [hv.Dimension("F0_x"), AngularSpace()],
                vdims = [FluorescenceAxis('Photons')],
            )
        else:
            f0HeatMap = hv.HeatMap(
                (
                    angular_axis,
                    np.zeros_like(f0s),
                    f0s
                ),
                kdims = [AngularSpace(), hv.Dimension("F0_x")],
                vdims = [FluorescenceAxis('Photons')],
            )
        f0HeatMap = f0HeatMap.opts(
            **opts
        )
        return f0HeatMap

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
        