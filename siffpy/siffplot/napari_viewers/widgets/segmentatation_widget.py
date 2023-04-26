"""
Creates a widget for identifying ROI extraction methods and their parameters,
plus a button for executing the extraction method + segmentation.
"""
from functools import wraps
import magicgui.widgets as widgets
import typing

from napari.layers import Shapes
from napari.utils.events import Event, EmitterGroup


from siffpy.siffplot.napari_viewers.napari_interface import NapariInterface
from siffpy.siffroi.roi_protocols.utils.napari_fcns import (
    rois_into_shapes_layer, PolygonSourceNapari
)
import siffpy.siffplot.napari_viewers.widgets.segmentation as segmentation
if typing.TYPE_CHECKING:
    from siffpy.siffroi.roi_protocols.roi_protocol import ROI, ROIProtocol


def fail_with_warning_window(func):
    @wraps(func)
    def wrapper(widget, *func_args):
        napari_interface = widget.napari_interface
        try:
            return func(widget, *func_args)
        except Exception as e:
            napari_interface.warning_window(
                f"An error occurred: {e}"
            )
    return wrapper

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
            segmented = Event,
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
        self.events.segmented.connect(self.refresh_subrois)
        
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
    def current_protocol(self)->'ROIProtocol':
        return self.primary_protocol.current_protocol

    @property
    def selected_rois(self)->list['ROI']:
        return self.current_rois_container.current_rois

    @property
    def extraction_kwargs(self)->dict[str, typing.Any]:
        return self.primary_protocol.extraction_kwargs

    @property
    def update_roi_list(self)->typing.Callable[[list['ROI']], None]:
        return self.current_rois_container.update_roi_list
    
    @property
    def segmentation_params(self)->dict[str, typing.Any]:
        return self.segmentation_params_container.segmentation_params

    @fail_with_warning_window
    def on_extraction_clicked(self, event):
        # This sends the extraction_initiated event to be called whenever
        # the protocol is done with its on_click job.
        self.current_protocol.on_click(self.events.extraction_initiated)

    @fail_with_warning_window
    def segmentation_callback(self, *args):
        self.events.segmented()

    @fail_with_warning_window
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

    @fail_with_warning_window
    def save_rois_callback(self):
        """ Calls the napari_interface's save_rois_fcn """
        if not hasattr(self.napari_interface, 'save_rois_fcn'):
            raise ValueError("No save rois function provided by viewer")

        if self.napari_interface.save_rois_fcn is None:
            raise ValueError("No save rois function implemented by viewer")

        self.napari_interface.save_rois_fcn()