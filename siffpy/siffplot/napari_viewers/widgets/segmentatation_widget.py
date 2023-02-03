"""
Creates a widget for identifying ROI extraction methods and their parameters,
plus a button for executing the extraction method + segmentation.
"""

import magicgui.widgets as widgets
import typing
import numpy as np
import inspect

from napari.layers import Shapes

from siffpy import SiffReader
from siffpy.siffplot.utils.exceptions import NoROIException
from siffpy.siffplot.roi_protocols import REGIONS, roi_protocol, ROI

from siffpy.siffplot.napari_viewers.napari_interface import NapariInterface
from siffpy.siffplot.roi_protocols.utils import PolygonSourceNapari
from siffpy.siffplot.roi_protocols.utils.napari_fcns import rois_into_shapes_layer
import siffpy.siffplot.napari_viewers.widgets.segmentation as segmentation


__all__ = ['create_segmentation_widget']

class SegmentationWidget(widgets.Container):
    def __init__(
        self,
        window : NapariInterface,
        segmentation_layer : Shapes,
        ):

        self.anatomical_region_box = segmentation.AnatomicalRegionBox(regions = REGIONS)
        
        
        starting_region = self.anatomical_region_box.current_choice
        region = next(region for region in REGIONS if region.region_enum.value == starting_region)
        starting_choices_str = list(fn.name for fn in region.functions)
        default_fcn = region.default_fcn
        # name of returned class
        starting_roi_class = typing.get_type_hints(region.default_fcn)['return']
        
        
        self.extract_method_box = segmentation.ExtractionMethodBox(
            starting_choices_str, region.default_fcn_str
        )

        self.roi_class = widgets.Label(
            name = "ROIClass",
            label = "ROI Class : ",
            value = starting_roi_class.__name__
        )

        self.extraction_params_container = segmentation.ExtractionParamsContainer(
            initial_method = region.default_fcn
        )
        
        self.segmentation_params_container = segmentation.SegmentationParamsContainer(
            initial_roi_class = starting_roi_class
        )
        self.segment_pushbutton = segmentation.SegmentPushbutton() 
        self.current_rois_container = segmentation.ROIsContainer(window)
        self.show_subrois = segmentation.ShowSubROIs()
        self.save_rois_pushbutton = segmentation.SaveRoisPushbutton()

        super().__init__(
            name='roi_widget_container',
            widgets=[
                self.anatomical_region_box,
                self.extract_method_box,
                self.roi_class,
                self.extraction_params_container,
                self.segmentation_params_container,
                self.segment_pushbutton,
                self.current_rois_container,
                self.show_subrois,
                self.save_rois_pushbutton,
                
            ],
            tooltip = "Tools for annotating regions of interest",
        )

        self.anatomical_region_box.connect_extraction_method_box(
            self.extract_method_box
        )

        self.extract_method_box.connect_other_widgets(
            self.anatomical_region_box,
            self.extraction_params_container,
            self.segmentation_params_container,
            self.roi_class,
        )


        self.max_width = 500
        
        self.segment_pushbutton.changed.connect(self.segmentation_callback)
        self.show_subrois.changed.connect(self.refresh_subrois)
        self.save_rois_pushbutton.changed.connect(self.save_rois_callback)

        self.window = window
        self.reference_frames = None
        self.segmentation_layer = segmentation_layer
        self.current_rois_container.refresh()

    def connect_reference_frames(self, reference_frames : np.ndarray):
        self.reference_frames = reference_frames

    def segmentation_callback(self, *args):
        try:
            if self.reference_frames is None:
                raise NoROIException("No reference frames loaded")
            region = self.anatomical_region_box.value
            method_name = self.extract_method_box.value

            #parse extraction params
            extraction_kwarg_dict = {}
            for widget in self.extraction_params_container:
                extraction_kwarg_dict[widget.name] = widget.value

            # See if any other args are needed for that method
            additional_args = []

            reg_obj = next((reg_obj for reg_obj in REGIONS if reg_obj.region_enum.value == region), None)
            func : typing.Callable = next((segfunc.func for segfunc in reg_obj.functions if segfunc.name == method_name), None)
            
            # Checks if it's a siffreader, because we know what to do with those
            if any(
                par.annotation == SiffReader
                for par in inspect.signature(func).parameters.values()
            ):
                additional_args.append(self.window.siffreader)

            rois = roi_protocol(
                region,
                method_name,
                self.reference_frames,
                PolygonSourceNapari(self.window.viewer),
                *additional_args,
                **extraction_kwarg_dict,
            )

            # run segmentation
            segmentation_kwarg_dict = {}
            for widget in self.segmentation_params_container:
                segmentation_kwarg_dict[widget.name] = widget.value

            roi : ROI
            for roi in rois:
                try:
                    roi.segment(**segmentation_kwarg_dict)
                except NotImplementedError:
                    pass

            visualizer = self.window.visualizer
            if hasattr(visualizer, 'rois'):
                if type(visualizer.rois) is list:
                    if not type(rois) is list:
                        visualizer.rois.append(rois)
                    else:
                        visualizer.rois += rois
                else:
                    if visualizer.rois is None:
                        visualizer.rois = rois
                    else:
                        visualizer.rois = [visualizer.rois, rois]
            else:
                if not type(rois) is list:
                    visualizer.rois = [rois]
                else:
                    visualizer.rois = rois
            
            if visualizer.rois is None:
                raise RuntimeError("No rois extracted -- check method used, images provided, etc.")

            self.current_rois_container.refresh()
            self.refresh_subrois(self.show_subrois.value)

        except Exception as e:
            self.window.warning_window(f"Error in segmentation function: {e}", exception = e)

    def refresh_subrois(self, checkbox_val : int):
        if not hasattr(self.window.visualizer, 'rois'):
            raise ValueError("No ROIs loaded")
        if self.window.visualizer.rois is None:
            raise ValueError("No ROIs loaded")
        
        if self.segmentation_layer is None:
            raise ValueError("No segmentation layer loaded")

        self.segmentation_layer.selected_data = set(range(self.segmentation_layer.nshapes))
        self.segmentation_layer.remove_selected()

        if not checkbox_val:
            return
        
        for roi in self.window.visualizer.rois:
            if hasattr(roi, 'subROIs'):
                self.segmentation_layer.visible = True
                for subroi in roi.subROIs:
                    if (not hasattr(subroi, 'slice_idx')) and hasattr(roi,'slice_idx'):
                        subroi.slice_idx = roi.slice_idx
                    rois_into_shapes_layer(subroi, self.segmentation_layer)

    def save_rois_callback(self):
        if not hasattr(self.window, 'save_rois_fcn'):
            raise ValueError("No save rois function provided by viewer")

        if self.window.save_rois_fcn is None:
            raise ValueError("No save rois function implemented by viewer")

        try:
            self.window.save_rois_fcn()

        except Exception as e:
            self.window.warning_window(f"Error in save ROI function: {e}", exception = e)
        