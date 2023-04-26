from typing import Any
import inspect
from inspect import Parameter
from enum import Enum

import magicgui.widgets as widgets

class SegmentationParamsContainer(widgets.Container):
    def __init__(self,
        initial_roi_class : type,
    ):
        super().__init__(
            name = "SegmentationContainer",
            label = "Segmentation\nparameters",
            layout = "vertical",
            tooltip = "Variables for segmentation of drawn ROIs."
        )
        self.current_roi_class = initial_roi_class
        self.refresh()

    def on_update_extraction_method(self, event):
        self.set_roi_class(event.source.current_protocol.return_class)

    def set_roi_class(self, new_roi_class : type):
        self.current_roi_class = new_roi_class
        self.refresh()

    @property
    def segmentation_params(self)->dict[str, Any]:
        """ Name and value of all parameters in the container."""
        return {widget.name: widget.value for widget in self}

    def refresh(self):
        """ Repopulates the parameters based on the current ROI class."""
        self.clear()
        if self.current_roi_class is None:
            return

        segment_fcn =  inspect.getmembers(
            self.current_roi_class,
            lambda x: inspect.isfunction(x) and x.__name__ == 'segment'
        )

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
                self.append(param_widget)

class SegmentPushbutton(widgets.PushButton):
    def __init__(self):
        super().__init__(
            name="SegmentPushButton",
            label="Segment selected ROI(s)",
            tooltip = "Constructs siffpy ROI objects using drawn shapes."
        )
