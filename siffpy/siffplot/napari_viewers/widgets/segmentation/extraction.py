import typing
from inspect import Parameter, signature, _empty
from enum import Enum

import magicgui.widgets as widgets

from siffpy.siffplot.roi_protocols import REGIONS

class ExtractionParamsContainer(widgets.Container):

    def __init__(self, initial_method : typing.Callable):
        super().__init__(
            name="ExtractionContainer",
            label="ROI extraction\nparameters",
            tooltip = "Parameters for primary ROI extraction"
        )
        self.current_method = initial_method
        self.refresh()

    def set_current_method(self, method : typing.Callable):
        self.current_method = method
        self.refresh()

    def refresh(self):
        self.clear()
        params = [
        kw for key, kw in signature(self.current_method).parameters.items()
            if kw.kind is Parameter.KEYWORD_ONLY
        ]
        for param in params:
            value = param.default
            param_widget = None
            if value is _empty:
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
        #self.append(extraction_method.params_widget)

class ExtractionMethodBox(widgets.ComboBox):
    def __init__(self, choices, value):
        super().__init__(
            name = 'ExtractionMethodBox',
            label='Extraction method',
            choices = choices,
            value = value,
            tooltip = "Which protocol to use to extract the ROI. For more information on each, TODO MAKE A USEFUL INFO TOOL, MAYBE A (?) icon"
        )

    def connect_other_widgets(
        self,
        anatomical_region_box,
        extraction_params_container : ExtractionParamsContainer,
        segmentation_params_container,
        roi_class_label,
    ):
        self.anatomical_region_box = anatomical_region_box
        self.extraction_params_container = extraction_params_container
        self.segmentation_params_container = segmentation_params_container
        self.roi_class_label = roi_class_label
        self.changed.connect(
            self.update_method_callback
        )

    def update_method_callback(self, method_name : str):
        region = next(
            x for x in REGIONS
            if self.anatomical_region_box.value in x.alias_list
        )
        method = next(
            x for x in region.functions
            if x.name == method_name
        )
        self.extraction_params_container.set_current_method(method.func)
        try:
            returned_class = typing.get_type_hints(method.func)['return']
        except KeyError:
            self.roi_class_label.value = "None (invalid function)"
            returned_class = None

        self.segmentation_params_container.set_roi_class(
            returned_class
        )

        try:
            self.roi_class_label.value = returned_class.__name__
        except:
            self.roi_class_label = "None (invalid function)"
            returned_class = None


    def update_region(self, region_name : str):
        region = next(x for x in REGIONS if region_name in x.alias_list)
        self.choices = list(fn.name for fn in region.functions)
        self.value = region.default_fcn_str