import functools, os, re, pickle, logging

import holoviews as hv
import numpy as np

from ...siffpy import SiffReader
from ..siffvisualizer import SiffVisualizer, apply_opts
from ..napari_viewers import ROIViewer
from .. import roi_protocols
from ..roi_protocols import rois
from ..utils import *

class ROIVisualizer(SiffVisualizer):
    """
    Extends the SiffVisualizer to provide
    annotation of reference frames to select
    ROIs.
    """
    def __init__(self, siffreader : SiffReader):
        super().__init__(siffreader)
        self.image_opts['clim'] = (0,1) # defaults to highest contrast

    @apply_opts
    def reference_frames_to_holomap(self)->hv.HoloMap:
        """
        If the current siffreader has an opened file,
        looks to see if there are stored reference frames,
        and if so returns a HoloViews HoloMap that allows
        viewing each of them
        """
        if not hasattr(self.siffreader, 'reference_frames'):
            logging.warning("No reference frames stored in siffreader")
            return None
        if self.siffreader.reference_frames is None:
            logging.warning("No reference frames stored in siffreader")
            return None
        
        self.ref_ds = hv.Dataset(
            (
                range(self.siffreader.im_params.xsize),
                range(self.siffreader.im_params.ysize), 
                range(self.siffreader.im_params.num_slices),
                self.siffreader.reference_frames
            ),
            ['x','y','z'], 'Intensity'
        )

        ref_holomap = self.ref_ds.to(hv.Image, ['x','y'], 'Intensity', groupby=['z'])
        # hard limits
        if hasattr(self,'local_opts'): # avoids initialization issues.
            ref_holomap = ref_holomap.opts(
                xlim = (0, self.siffreader.im_params.xsize),
                ylim = (0, self.siffreader.im_params.ysize)
            )
            if not self.local_opts is None:
                ref_holomap = ref_holomap.opts(*self.local_opts)
        return ref_holomap


    def get_roi_reference_layouts(self, merge : bool = True, polygon_shape : str = 'polygons', **kwargs) -> dict[int, dict]:
        """
        TODO: Maybe don't overwrite existing ROIs?
        
        Returns a dict of dicts, the structure of which is as follows:

        dict = {
            z_index (int) :
                {
                    'annotator' : hv.annotator instance for z plane,
                    'layout'    : hv.Layout instance showing the reference frame and polygons
                }
            for z_index in range(num_slices)
        }

        KEYWORD ARGUMENTS
        -----------------

        merge : bool (default True)

            Adds a merged Holoviews Layout object that contains all the reference planes in one Layout,
            each with their own Bokeh toolbar. The key in the returned dict is 'merged', not an int

        polygon_shape : str (default is 'polygons')

            Shape of the ROIs being drawn. Options are:
                polygons
                rectangles
                ellipses
        """
        if not hasattr(self, 'reference_frames'):
            self.reference_frames = self.reference_frames_to_holomap()

        # this is kind of a silly construction, but I think it's
        # more clear to spell everything out little bit by little bit
        # this way. 
        if re.match(r'polygon[s]?', polygon_shape, re.IGNORECASE):
            drawdict = {
                zidx:
                    hv.Polygons([])
                for zidx in range(self.siffreader.im_params.num_slices)
            }
        elif re.match(r'rectangle[s]?', polygon_shape, re.IGNORECASE):
            drawdict = {
                zidx:
                    hv.Rectangles([])
                for zidx in range(self.siffreader.im_params.num_slices)
            }
        elif re.match(r'ellipse[s]?', polygon_shape, re.IGNORECASE):
            raise NotImplementedError("Ellipse keyword argument not yet implemented (not a native Hv/Bokeh drawer)")
        else:
            raise ValueError(f"Invalid optional argument for polygon_shape:\n\t{polygon_shape}")

        annotators = {
            zidx :
                hv.annotate.instance()
            for zidx in range(self.siffreader.im_params.num_slices)
        }

        annotator_layouts = {
            zidx:
                annotator(self.reference_frames[zidx] * drawdict[zidx]).opts(
                    hv.opts.Table(width=0), # hide the annotation table
                    hv.opts.Layout(merge_tools=False) # don't share the tools
                )
            for zidx, annotator in annotators.items()
        }

        self.annotation_dict = {
            zidx: {
                'annotator' : annotators[zidx],
                'layout'    : annotator_layouts[zidx]
            }
            for zidx in annotators.keys()
        }

        if merge:
            # Merges into a single layout with toolbars for each holoviews element
            def merge_plots(a, b):
                if isinstance(a, dict):
                    return (a['layout'] + b['layout']).opts(merge_tools = False)
                else:
                    return (a+b['layout']).opts(merge_tools=False)

            self.annotation_dict['merged'] =  functools.reduce(merge_plots, list(self.annotation_dict.values()))

        return self.annotation_dict

    def draw_rois(self, keep_old : bool = False, **kwargs):
        """
        Creates a GUI that
        enables drawing and editing polygons
        on each of the reference frames for the
        opened .siff file. When the appropriate
        annotation has been performed, you can
        call extract_rois (or anything else that
        depends on the annotation).

        Arguments
        ---------

        keep_old : bool (optional)

            If True, retains other ROIs already drawn for this
            SiffPlotter object, as well as any new ones drawn
            here (which means they'll also get fed into the 
            extract_rois call). Default is False.

        Holoviews backend accepts all keyword arguments of get_roi_reference_layouts.

        Returns
        -------

        - If using napari:

        viewer : napari.Viewer

            A napari Viewer object with a few initial layers: a
            Shapes layer and an Image layer. The Image layer shows
            the reference frames of the SiffReader attribute.

        - If using HoloViews:

        self.annotation_dict['merged'] : holoviews.Layout

            A pointer to the SiffPlotter's merged annotation Layout
            that shows all the reference frames fused together into
            a single layout, accompanied by Holoviews Annotator
            objects to track, for example, added polygons.
        """
        if self.backend == 'napari':
            return self.draw_rois_napari(**kwargs)
        else:
            return self.draw_rois_hv(keep_old = keep_old, **kwargs)

    def draw_rois_napari(self, **kwargs)->None:
        """
        Returns a napari Viewer object that shows
        the reference frames of the .siff file and
        a layer for overlaying drawn polygons and shapes.
        """
        if not hasattr(self.siffreader, 'reference_frames'):
            raise AssertionError("SiffReader has no registered reference frames.")

        self.viewer = ROIViewer(self.siffreader, visualizer = self, title='Annotate ROIs')
        self.viewer.save_rois_fcn = self.save_rois


    def draw_rois_hv(self, keep_old : bool = False, **kwargs)->hv.Layout:
        """
        Returns a hv.Layout element that
        enables drawing and editing polygons
        on each of the reference frames for the
        opened .siff file. When the appropriate
        annotation has been performed, you can
        call extract_rois (or anything else that
        depends on the annotation).

        Arguments
        ---------

        keep_old : bool (optional)

            If True, retains other ROIs already drawn for this
            SiffPlotter object, as well as any new ones drawn
            here (which means they'll also get fed into the 
            extract_rois call). Default is False.

        Accepts all keyword arguments of get_roi_reference_layouts.

        Returns
        -------

        self.annotation_dict['merged'] : holoviews.Layout

            A pointer to the SiffPlotter's merged annotation Layout
            that shows all the reference frames fused together into
            a single layout, accompanied by Holoviews Annotator
            objects to track, for example, added polygons.
        """
        # If there is no annotation_dict or merged
        # version, then override whatever the kwarg
        # is telling you to do.
        if not hasattr(self, 'annotation_dict'):
            keep_old = False
        else:
            if not 'merged' in self.annotation_dict:
                keep_old = False
        
        if not keep_old:
            self.get_roi_reference_layouts(merge = True, **kwargs)
        
        return self.annotation_dict['merged']

    def select_points(self, roi : rois.ROI , **kwargs)->hv.Overlay:
        """
        Allows selection of points on an ROI polygon with HoloViews.

        Arguments
        ---------

        roi : siffplot.roi_protocols.rois.ROI

            The ROI to annotate

        Returns
        -------

        selection_field : hv.Overlay

            Returns an overlay of the ROI(s), the reference frame (if linked to the ROI), and a DynamicMap that can be used to select points
            on the ROI(s). Selected points are stored by the ROI, so when you save the ROI they will have the points too!

        """
        # First check that the needed information is present somewhere.
        if (self.backend == 'napari'):
            raise AttributeError("SiffVisualizer object is using the napari backend, not HoloViews.")

        if not hasattr(self,'annotation_dict'):
            raise AssertionError("No annotated reference frames provided for this SiffPlotter. Try calling draw_rois() or get_roi_reference_layouts() first.")

        if not hasattr(roi,'selected_points'):
            selected_points = []
            roi.selected_points = selected_points
        else:
            selected_points = roi.selected_points

        points_dict = roi.polygon.data[0]
        
        points_array = np.array(list(zip(points_dict['x'],points_dict['y']))) # To make the function easier.

        tap_fn = lambda x, y, x2, y2 : select_on_tap(points_array, selected_points, x, y, x2, y2) # defined in siffutils

        # Create two streams into one DynamicMap
        tap_stream = hv.streams.SingleTap(transient=True , x=None, y = None)        
        dtap_stream = hv.streams.DoubleTap(rename={'x': 'x2', 'y': 'y2'}, transient=True, x=None, y=None)
            
        tap_dmap = hv.DynamicMap(tap_fn, streams=[tap_stream, dtap_stream]).opts(fill_color="#FF0000", size=10)

        if hasattr(roi, 'slice_idx'):
            return self.reference_frames[roi.slice_idx] * roi.polygon * tap_dmap
        return roi.polygon * tap_dmap

    def roi_to_layout(self):
        """
        Should've written a docstring.
        Can't remember what this was supposed to be for.
        But I suspect I'll want it later.
        """
        raise NotImplementedError("")

    def extract_rois(self, region : str, *args, method_name : str = None, overwrite : bool = True, **kwargs) -> None:
        """
        Extract ROIs -- uses a different method for each anatomical region.
        ROIs are stored in a class attribute. Must have drawn at least one
        manual ROI on at least one image stored in the SiffPlotter's annotation dict
        or in a napari Viewer generated by this SiffPlotter.

        To learn more about 

        Parameters
        ----------
        
        region : str

            Name of the region of interest. Current protocols are for the FB, EB, and PB.

        method_name : str (optional)

            Which ROI extraction method to use. For a list, call siffplot.ROI_fitting_methods()

        overwrite : bool (optional)

            If set to True, overwrites self.rois rather than appending to it. Default is True.
        
        Returns
        -------

        None

        """

        if not (self.backend == 'napari'):
            if not hasattr(self,'annotation_dict'):
                raise AssertionError("No annotators generated yet. Try draw_rois()")
            if not any([len(x['annotator'].annotated.data) for x in self.annotation_dict.values() if isinstance(x,dict)]):
                raise AssertionError("ROIVisualizer object has no hand-annotated ROIs")
        else:
            if not hasattr(self,'viewer'):
                raise AssertionError("No associated napari viewer. Try draw_rois()!")

        self.region = roi_protocols.region_name_proper(region)
        
        if (self.backend == 'napari'):
            # Make sure the napari viewer has a layer named ROI viewer
            try:
                filter(lambda layer: layer.name == 'ROI shapes', self.viewer.layers),            
            except Exception:
                raise AssertionError("Failed to identify an ROI shapes layer.")

            rois = roi_protocols.roi_protocol(
            region,
            method_name,
            self.siffreader.reference_frames,
            self.viewer,
            **kwargs
        )
        else:     
            rois = roi_protocols.roi_protocol(
                region,
                method_name,
                self.siffreader.reference_frames,
                self.annotation_dict,
                **kwargs
            )
        
        # Now we have the ROIs, time to
        # make a reference to them that
        # survives so they won't be garbage
        # collected.

        if overwrite:
            # overwrites the current set
            # and returns
            if not type(rois) is list:
                self.rois = [rois]
            else:
                self.rois = rois
            return

        if hasattr(self, 'rois'):
            if type(self.rois) is list:
                if not type(rois) is list:
                    self.rois.append(rois)
                else:
                    self.rois += rois
            else:
                self.rois = [self.rois, rois]
        else:
            if not type(rois) is list:
                self.rois = [rois]
            else:
                self.rois = rois
        
        if self.rois is None:
            raise RuntimeError("No rois extracted -- check method used, images provided, etc.")
    
    def redraw_rois(self):
        """ Redraws the rois, for example after segmentation. """
        raise NotImplementedError()

    def __getattribute__(self, name: str):
        """
        To make it easier to access when there's only one ROI
        (there's something gross about having to have a bunch of [0]
        sitting around in your code)
        """
        if name == 'rois':
            roi_ref = object.__getattribute__(self, name)
            if type(roi_ref) is list:
                if len(roi_ref) == 1:
                    return roi_ref[0]
            return roi_ref
        else:
            return object.__getattribute__(self, name)
