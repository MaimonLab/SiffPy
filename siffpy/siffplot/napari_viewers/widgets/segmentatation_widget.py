"""
Creates a widget for identifying ROI extraction methods and their parameters,
plus a button for executing the extraction method + segmentation.
"""

import magicgui.widgets as widgets
import typing

from napari.layers import Shapes
from napari.utils.events import Event, EventEmitter, EmitterGroup


from siffpy.siffplot.napari_viewers.napari_interface import NapariInterface
from siffpy.siffplot.roi_protocols.utils.napari_fcns import (
    rois_into_shapes_layer, PolygonSourceNapari
)
import siffpy.siffplot.napari_viewers.widgets.segmentation as segmentation


__all__ = ['create_segmentation_widget']

class SegmentationWidget(widgets.Container):
    def __init__(
        self,
        napari_interface : NapariInterface,
        segmentation_layer : Shapes,
        ):

        self.events = EmitterGroup(
            source=self,
        )
        self.events.add(
            extraction_initiated = Event,
            extraction_complete = Event,
            update_rois = Event,
            segment = Event,
        )

        ### ROI extraction widgets
        self.primary_protocol = segmentation.PrimaryProtocol() 
        
        self.primary_protocol.extraction_button_clicked.connect(
            self.on_extraction_clicked
        )

        ## Current ROI container
        self.current_rois_container = segmentation.ROIsContainer()

        ### SEGMENTATION widgets
        self.segmentation_params_container = segmentation.SegmentationParamsContainer(
            initial_roi_class = self.primary_protocol.current_protocol.return_class
        )
        self.segment_pushbutton = segmentation.SegmentPushbutton() 
        self.show_subrois = segmentation.ShowSubROIs()

        self.segment_pushbutton.changed.connect(self.segmentation_callback)
        self.show_subrois.changed.connect(self.refresh_subrois)
        
        #This SHOULD hook up to the ROI container
        self.primary_protocol.updated_method.connect(
            self.segmentation_params_container.on_update_extraction_method
        )

        #SAVE button
        self.save_rois_pushbutton = segmentation.SaveRoisPushbutton()
        self.save_rois_pushbutton.changed.connect(self.save_rois_callback)

        super().__init__(
            name='roi_widget_container',
            widgets=[
                *self.primary_protocol.widgets,
                self.current_rois_container,
                self.segmentation_params_container,
                self.segment_pushbutton,
                self.show_subrois,
                self.save_rois_pushbutton,
            ],
            tooltip = "Tools for annotating regions of interest",
        )

        self.max_width = 500
        
        self.napari_interface = napari_interface
        self.segmentation_layer = segmentation_layer

    @property
    def current_protocol(self):
        return self.primary_protocol.current_protocol

    @property
    def extraction_kwargs(self)->dict:
        return self.primary_protocol.extraction_kwargs

    @property
    def update_roi_list(self)->typing.Callable:
        return self.current_rois_container.update_roi_list

    def on_extraction_clicked(self, event):
        # This sends the extraction_initiated event to be called whenever
        # the protocol is done with its on_click job.
        self.current_protocol.on_click(self.events.extraction_initiated)

    def segmentation_callback(self, *args):
        self.napari_interface.warning_window(
            'Segmentation in process of being re-implemented!'
        )
        pass

        # try:
        #     if self.reference_frames is None:
        #         raise NoROIException("No reference frames loaded")
        #     region = self.primary_protocol.current_region
        #     method_name = self.primary_protocol.extract_method_box.value

        #     #parse extraction params
        #     extraction_kwarg_dict = {}
        #     for widget in self.extraction_params_container:
        #         extraction_kwarg_dict[widget.name] = widget.value

        #     # See if any other args are needed for that method
        #     additional_args = []

        #     reg_obj = next((reg_obj for reg_obj in REGIONS if reg_obj.region_enum.value == region), None)
        #     func : typing.Callable = next((segfunc.func for segfunc in reg_obj.functions if segfunc.name == method_name), None)
            
        #     # Checks if it's a siffreader, because we know what to do with those
        #     if any(
        #         par.annotation == SiffReader
        #         for par in inspect.signature(func).parameters.values()
        #     ):
        #         additional_args.append(self.napari_interface.siffreader)

        #     rois = roi_protocol(
        #         region,
        #         method_name,
        #         self.reference_frames,
        #         PolygonSourceNapari(self.napari_interface.viewer),
        #         *additional_args,
        #         **extraction_kwarg_dict,
        #     )

        #     # run segmentation
        #     segmentation_kwarg_dict = {}
        #     for widget in self.segmentation_params_container:
        #         segmentation_kwarg_dict[widget.name] = widget.value

        #     roi : ROI
        #     for roi in rois:
        #         try:
        #             roi.segment(**segmentation_kwarg_dict)
        #         except NotImplementedError:
        #             pass

        #     visualizer = self.napari_interface.visualizer
        #     if hasattr(visualizer, 'rois'):
        #         if type(visualizer.rois) is list:
        #             if not type(rois) is list:
        #                 visualizer.rois.append(rois)
        #             else:
        #                 visualizer.rois += rois
        #         else:
        #             if visualizer.rois is None:
        #                 visualizer.rois = rois
        #             else:
        #                 visualizer.rois = [visualizer.rois, rois]
        #     else:
        #         if not type(rois) is list:
        #             visualizer.rois = [rois]
        #         else:
        #             visualizer.rois = rois
            
        #     if visualizer.rois is None:
        #         raise RuntimeError("No rois extracted -- check method used, images provided, etc.")

        #     self.current_rois_container.refresh(self.napari_interface)
        #     self.refresh_subrois(self.show_subrois.value)

        # except Exception as e:
        #     self.napari_interface.warning_window(f"Error in segmentation function: {e}", exception = e)

    def refresh_subrois(self, checkbox_val : int):
        if not hasattr(self.napari_interface.visualizer, 'rois'):
            raise ValueError("No ROIs loaded")
        if self.napari_interface.visualizer.rois is None:
            raise ValueError("No ROIs loaded")
        
        if self.segmentation_layer is None:
            raise ValueError("No segmentation layer loaded")

        self.segmentation_layer.selected_data = set(range(self.segmentation_layer.nshapes))
        self.segmentation_layer.remove_selected()

        if not checkbox_val:
            return
        
        for roi in self.napari_interface.visualizer.rois:
            if hasattr(roi, 'subROIs'):
                self.segmentation_layer.visible = True
                for subroi in roi.subROIs:
                    if (not hasattr(subroi, 'slice_idx')) and hasattr(roi,'slice_idx'):
                        subroi.slice_idx = roi.slice_idx
                    rois_into_shapes_layer(subroi, self.segmentation_layer)

    def save_rois_callback(self):
        if not hasattr(self.napari_interface, 'save_rois_fcn'):
            raise ValueError("No save rois function provided by viewer")

        if self.napari_interface.save_rois_fcn is None:
            raise ValueError("No save rois function implemented by viewer")

        try:
            self.napari_interface.save_rois_fcn()

        except Exception as e:
            self.napari_interface.warning_window(f"Error in save ROI function: {e}", exception = e)
        