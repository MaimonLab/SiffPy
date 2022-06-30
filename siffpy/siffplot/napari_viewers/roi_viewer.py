from enum import Enum
import typing, inspect
from typing import Callable
from inspect import Parameter

import numpy as np
import magicgui.widgets as widgets

from .napari_interface import NapariInterface
from ...siffpy import SiffReader
from ..utils.exceptions import NoROIException
from ..roi_protocols import ROI_extraction_methods, REGIONS, roi_protocol
from ..roi_protocols.rois import ROI
from ..roi_protocols.utils import napari_fcns

CINNABAR = '#db544b'
DRAWN_SHAPE_LAYER_NAME = "ROI shapes"
SUBROI_LAYER_NAME = "Segmented ROIs"

class ROIViewer(NapariInterface):
    """
    Access to a napari Viewer object specialized for annotating ROIs.
    Designed to behave LIKE a Viewer without subclassing the Viewer
    directly.

    TODO: FINISH IMPLEMENTING. Most important features: 

    -- Fix the subROI segmentation function when it happens multiple
    times

    -- Allow selecting ROIs from the side panel and deleting them
    (or highlighting them)
    """

    def __init__(self, siffreader : SiffReader, *args, segmentation_fcn = None, edge_color = CINNABAR, **kwargs):
        """
        Accepts all napari.Viewer arguments plus requires a siffpy.SiffReader
        object as its first argument.

        Accepts napari.Viewer args and keyword arguments in addition to the below:

        Parameters
        -----------


        Keyword arguments
        -----------------

        segmentation_fcn : Callable

            Function called by Segment ROIs button.

        edge_color : str (hex color code)

            Color of drawn ROI edges

        """
        super().__init__(siffreader, *args, **kwargs)
        self.viewer.dims.axis_labels = ['Z planes', 'x', 'y']

        self.segmentation_fcn = segmentation_fcn
        self.save_rois_fcn : Callable = None

        self.initialize_layers(edge_color = edge_color)
        self.initialize_segmentation_widget()
    
    def initialize_layers(self, edge_color = CINNABAR):
        """ Initializes the napari viewer layer for drawing ROIs """
        
        if not hasattr(self.siffreader,'reference_frames'):
            raise NoROIException("SiffReader has no reference frames")

        self.viewer.add_image(
            data = np.array(self.siffreader.reference_frames),
            name='Reference frames',
            scale = self.scale,
        )

        self.viewer.add_shapes(
            face_color="transparent",
            name=DRAWN_SHAPE_LAYER_NAME,
            ndim=3,
            edge_color=edge_color,
            scale = self.scale
        )

        self.add_roi_object_layer()

        self.viewer.add_shapes(
            face_color = "transparent",
            name = SUBROI_LAYER_NAME,
            ndim = 3,
            edge_color = "#FFFFFF",
            scale = self.scale,
            visible = False,
            opacity = 0.3
        )

    def initialize_segmentation_widget(self):
        """ Initializes the part of the GUI for segmentation of drawn ROIs """

        anatomical_region_box = widgets.ComboBox(
            name = 'AnatomicalRegionBox',
            label='Anatomical region',
            choices = REGIONS.keys(),
            tooltip = 'The anatomical region determines which segmentation protocols can be used'
        )

        starting_region = anatomical_region_box.current_choice

        starting_choices = ROI_extraction_methods(print_output=False)[starting_region]
        default_fcn = REGIONS[starting_region]['default_fcn']

        # name of returned class
        starting_roi_class = typing.get_type_hints(
            inspect.getmembers(REGIONS[starting_region]['module'], lambda x: inspect.isfunction(x) and x.__name__ == default_fcn)[0][1]
        )['return']

        extraction_method_box = widgets.ComboBox(
            name = 'ExtractionMethodBox',
            label='Extraction method',
            choices = starting_choices,
            value = default_fcn,
            tooltip = "Which protocol to use to extract the ROI. For more information on each, TODO MAKE A USEFUL INFO TOOL, MAYBE A (?) icon"
        )

        roi_class = widgets.Label(
            name = "ROIClass",
            label = "ROI Class : ",
            value = starting_roi_class.__name__
        )

        extraction_params_container = widgets.Container(
            name="ExtractionContainer",
            label="ROI extraction\nparameters",
            tooltip = "Parameters for primary ROI extraction"
        )

        self.extraction_params_container = extraction_params_container

        segmentation_params_container = widgets.Container(
            name = "SegmentationContainer",
            label = "Segmentation\nparameters",
            layout = "vertical",
            tooltip = "Variables for segmentation of drawn ROIs."
        )

        self.segmentation_params_container = segmentation_params_container

        segment_pushbutton = widgets.PushButton(
            name="SegmentPushButton",
            label="Segment ROIs",
            tooltip = "Constructs siffpy ROI objects using drawn shapes.")

        current_rois_container = widgets.Container(
            name = "CurrentROIsContainer",
            layout = 'horizontal',
            label = "Current ROIs: ",
            tooltip = "ROIs stored in the SiffVisualizer's ROIs attribute.",
            visible = False
        )

        show_subrois = widgets.CheckBox(
            name = "ShowSubROIsBox",
            label = "Show sub-ROIs",
            tooltip = "Toggles whether sub-ROI masks are shown on the Napari viewer"
        )

        save_rois_pushbutton = widgets.PushButton(
            name="SaveRoiPushButton",
            label = "Save ROIs",
            tooltip = "Saves current ROIs stored in SiffViewer object"
        )
                                        
        roi_widget = widgets.Container(
            name='roi_widget_container',
            widgets=[
                anatomical_region_box, 
                extraction_method_box, 
                roi_class,
                extraction_params_container,
                segmentation_params_container,
                segment_pushbutton,
                current_rois_container,
                show_subrois,
                save_rois_pushbutton,
            ],
            tooltip = "Tools for annotating regions of interest",
        )

        roi_widget.max_width = 500

        # EVENTS AND CALLBACKS

        # WARNING: these are all stateful! even if not referenced by self directly.
        # they all have references from the Container widgets.

        def populate_extraction_params(extraction_params_container : widgets.Container, extraction_method : str):
            """ Updates the parameters for the extraction method by parsing the Python code for the extraction function"""
            extraction_params_container.clear()
            new_module = REGIONS[anatomical_region_box.value]['module']
            try:
                method_itself = next(x[1] for x in inspect.getmembers(new_module, inspect.isfunction) if x[0] == extraction_method)
            except Exception as e:
                return
            
            params = [
                kw for key, kw in inspect.signature(method_itself).parameters.items()
                if kw.kind is Parameter.KEYWORD_ONLY
            ]

            for param in params:
                value = param.default
                param_widget = None
                if value is inspect._empty:
                    value = None
                if param.annotation is str:
                    param_widget = widgets.LineEdit(
                        name = f"{param.name}",
                        label = param.name,
                        value = value
                    )
                if issubclass(param.annotation,Enum):
                    param_widget = widgets.ComboBox(
                        name = f"{param.name}",
                        label = param.name,
                        choices = [option.value for option in type(value)],
                        value = value.value
                    )
                if param.annotation is int:
                    param_widget = widgets.SpinBox(
                        name = f"{param.name}",
                        label = param.name,
                        value = value
                    )
                if param.annotation is float:
                    param_widget = widgets.FloatSpinBox(
                        name = f"{param.name}",
                        label = param.name,
                        value = value
                    )
                if not param_widget is None:
                    extraction_params_container.append(param_widget)

        def populate_segmentation_params(segmentation_params_container : widgets.Container, roi_class : type):
            """ Updates the parameters for the segmentation method by parsing the Python code for the segment function """
            segmentation_params_container.clear()
            if roi_class is None:
                return
            
            segment_fcn = inspect.getmembers(roi_class, lambda x: inspect.isfunction(x) and x.__name__ == 'segment')
            if len(segment_fcn) == 0:
                return
            
            segment_fcn = segment_fcn[0][1]

            seg_params = [val for key, val in inspect.signature(segment_fcn).parameters.items()
                if (not (key == 'self') and val.kind == Parameter.POSITIONAL_OR_KEYWORD)
            ]            
            if len(seg_params) == 0:
                return

            # Now the more generic approach
            for param in seg_params:
                value = param.default
                param_widget = None
                if value is inspect._empty:
                    value = None
                if param.annotation is str:
                    param_widget = widgets.LineEdit(
                        name = f"{param.name}",
                        label = param.name,
                        value = value
                    )
                if issubclass(param.annotation,Enum):
                    param_widget = widgets.ComboBox(
                        name = f"{param.name}",
                        label = param.name,
                        choices = [option.value for option in type(value)],
                        value = value.value
                    )
                if param.annotation is int:
                    param_widget = widgets.SpinBox(
                        name = f"{param.name}",
                        label = param.name,
                        value = value
                    )
                if param.annotation is float:
                    param_widget = widgets.FloatSpinBox(
                        name = f"{param.name}",
                        label = param.name,
                        value = value
                    )
                if not param_widget is None:
                    segmentation_params_container.append(param_widget)

        def update_region(region_name : str):
            """ Update callback when the selected brain region is changed. """
            extraction_method_box.choices = ROI_extraction_methods(print_output=False)[region_name]
            extraction_method_box.value = REGIONS[region_name]['default_fcn']

        def update_extraction_method(method_name : str):
            """ Updates the parameters for the extraction method """
            populate_extraction_params(extraction_params_container, method_name)
            update_segmentation_method(method_name)

        def update_segmentation_method(method_name : str):
            """ Updates the string naming the class of the returned ROI and the roi_widget container """
            new_module = REGIONS[anatomical_region_box.value]['module']
            try:
                method_itself = next(x[1] for x in inspect.getmembers(new_module, inspect.isfunction) if x[0] == method_name)
            except Exception as e:
                return
            try:
                returned_class = typing.get_type_hints(method_itself)['return']
            except KeyError:
                roi_class.value = "None (invalid function)"
                populate_segmentation_params(segmentation_params_container, None)
                return
            roi_class.value = returned_class.__name__
            populate_segmentation_params(segmentation_params_container, returned_class)

        self.populate_rois_container(current_rois_container, self.visualizer)
        populate_extraction_params(extraction_params_container, default_fcn)
        populate_segmentation_params(segmentation_params_container, starting_roi_class)
            
        anatomical_region_box.changed.connect(update_region)
        extraction_method_box.changed.connect(update_extraction_method)
        segment_pushbutton.changed.connect(self.segmentation_callback)
        show_subrois.changed.connect(self.subROI_check_callback)
        save_rois_pushbutton.changed.connect(self.save_rois_callback)

        self.roi_widgets = roi_widget
        self.viewer.window.add_dock_widget(self.roi_widgets, name='ROI segmentation tools')

    @property
    def segmented_rois_layer(self):
        """ May not exist for all viewers or even may be deleted for some reason. """
        roi_object_layer = next(
            filter(
                    lambda x: x.name == SUBROI_LAYER_NAME,
                    self.viewer.layers
                ),
            None
        )
        return roi_object_layer

    def populate_rois_container(self, roi_container : widgets.Container, visualizer):
            """ Provides info on existing rois stored by the accompanying visualizer object. """
            try:
                self.draw_rois_on_napari()
            except NoROIException:
                return
            
            roi_container.clear()
            roi_container.visible = False
            if self.visualizer is None:
                return
            if not hasattr(self.visualizer, 'rois'):
                return
            if self.visualizer.rois is None:
                return
            if len(self.visualizer.rois) == 0:
                return
            
            

            # label the top row
            names = widgets.Container(
                layout='vertical',
                widgets = [
                    widgets.Label(label='ROI name'),
                ],
            )

            classes = widgets.Container(
                layout='vertical',
                widgets = [widgets.Label(label='ROI type')],
            )

            segmenteds = widgets.Container(
                layout='vertical',
                widgets = [
                    widgets.Label(label="Segmented?")
                ],
            )

            nsegs = widgets.Container(
                layout='vertical',
                widgets = [widgets.Label(label="N segments?")],
            )


            roi_container.extend([names, classes, segmenteds, nsegs])
            
            roi : ROI # type hinting
            for roi in self.visualizer.rois:
                names.append(widgets.Label(label=roi.name))
                classes.append(widgets.Label(label=type(roi).__name__))
                if hasattr(roi,'subROIs'):
                    segmenteds.append(widgets.Label(label="Y"))
                    nsegs.append(widgets.Label(label=f"{len(roi.subROIs)}"))
                else:
                    segmenteds.append(widgets.Label(label="N"))
                    nsegs.append(widgets.Label(label="0"))

            roi_container.visible = True

    def segmentation_callback(self, pushbutton_val : int):
        try:
            region = self._widget_val('AnatomicalRegionBox')
            method_name = self._widget_val('ExtractionMethodBox')
            # parse each extraction param
            extraction_kwarg_dict = {}
            for widget in self.extraction_params_container:
                extraction_kwarg_dict[widget.name] = widget.value

            # Run the roi protocol
            rois = roi_protocol(
                region,
                method_name,
                self.siffreader.reference_frames,
                self,
                **extraction_kwarg_dict
            )

            # Run the segmentation protocol
            segmentation_kwarg_dict = {}
            for widget in self.segmentation_params_container:
                segmentation_kwarg_dict[widget.name] = widget.value

            for roi in rois:
                roi.segment(**segmentation_kwarg_dict)

            if hasattr(self.visualizer, 'rois'):
                if type(self.visualizer.rois) is list:
                    if not type(rois) is list:
                        self.visualizer.rois.append(rois)
                    else:
                        self.visualizer.rois += rois
                else:
                    if self.visualizer.rois is None:
                        self.visualizer.rois = rois
                    else:
                        self.visualizer.rois = [self.visualizer.rois, rois]
            else:
                if not type(rois) is list:
                    self.visualizer.rois = [rois]
                else:
                    self.visualizer.rois = rois
            
            if self.visualizer.rois is None:
                raise RuntimeError("No rois extracted -- check method used, images provided, etc.")

            self.populate_rois_container(self.roi_widgets['CurrentROIsContainer'], self.visualizer)
            self.subROI_check_callback(self.roi_widgets['ShowSubROIsBox'].value)

        except Exception as e:
            self.warning_window(f"Error in segmentation function: {e}", exception = e)
        
    def _widget_val(self, widg_name : str):
        """ Keeps .__dict__ small but lets me store lots of widgets """
        try:
            widget = next(x for x in self.roi_widgets._list if x.name == widg_name)
            if hasattr(widget,'value'):
                return widget.value
        except:
            return None
        

    def subROI_check_callback(self, checkbox_val : int) :
        """ Called on subROI box check"""
        if not hasattr(self.visualizer, 'rois'):
            raise ValueError("No ROIs loaded in SiffVisualizer")
        if self.visualizer.rois is None:
            raise ValueError("No ROIs loaded in SiffVisualizer")

        if self.segmented_rois_layer is None:
            return
        
        # clear the layer
        self.segmented_rois_layer.selected_data = set(range(self.segmented_rois_layer.nshapes))
        self.segmented_rois_layer.remove_selected()
        
        if not checkbox_val:
            return

        for roi in self.visualizer.rois:
            if hasattr(roi,'subROIs'):
                self.segmented_rois_layer.visible = True
                for subroi in roi.subROIs:
                    if (not hasattr(subroi, 'slice_idx')) and hasattr(roi,'slice_idx'):
                        subroi.slice_idx = roi.slice_idx
                    napari_fcns.rois_into_shapes_layer(subroi, self.segmented_rois_layer)


    def save_rois_callback(self, save_rois_val : int) : 
        """ Called when Save ROIs button is pushed """

        try:
            if self.save_rois_fcn is None:
                raise ValueError("No save_rois function implemented.")
            else:
                self.save_rois_fcn()
        except Exception as e:
            self.warning_window(f"Error in save ROI function: {e}", exception = e)
