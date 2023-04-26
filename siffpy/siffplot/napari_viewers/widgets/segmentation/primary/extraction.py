from inspect import _empty, getdoc
from enum import Enum

import magicgui.widgets as widgets
from napari.utils.events import Event, EventEmitter, EmitterGroup

from siffpy.siffroi.roi_protocols import REGIONS
from siffpy.siffroi.roi_protocols.roi_protocol import ROIProtocol

class ExtractionParamsContainer(widgets.Container):
    """
    The Widget storage for ROI extraction methods.
    """

    def __init__(self, initial_protocol : ROIProtocol):
        super().__init__(
            name="ExtractionContainer",
            label="ROI extraction\nparameters",
            tooltip = "Parameters for primary ROI extraction"
        )
        self.current_protocol = initial_protocol

        self.param_widget_container = widgets.Container(
            name = "ParamWidgetContainer",
            label = "Parameters",
            tooltip = "Parameters for primary ROI extraction",
            layout="vertical",
        )

        self.extract_pushbutton = widgets.PushButton(
            text = self.current_protocol.base_roi_text,
            name = "ExtractPushbutton",
            tooltip = "Extract the ROI",
        )
        self.events = EmitterGroup(
            source=self,
        )

        self.events.add(
            extraction_button_clicked=Event,
        )

        self.extract_pushbutton.clicked.connect(self.on_extract)

        self.extend(
            [self.param_widget_container,self.extract_pushbutton]
        )

        self.refresh()

    def on_extract(self, event):
        self.events.extraction_button_clicked()

    def set_current_method(self, protocol : ROIProtocol):
        self.current_protocol = protocol
        self.refresh()

    def refresh(self):
        self.param_widget_container.clear()

        params = [
            kw for key, kw in self.current_protocol.extraction_args.items()
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
                self.param_widget_container.append(param_widget)
        self.extract_pushbutton.text = self.current_protocol.base_roi_text
        
    def on_update_method(self, event : Event):
        self.set_current_method(event.source.current_protocol)

    def to_dict(self):
        return {
            widget.name : widget.value
            for widget in self.param_widget_container
        }

class ExtractionMethodBox(widgets.ComboBox):
    def __init__(self, choices, value):
        super().__init__(
            name = 'ExtractionMethodBox',
            label='Extraction method',
            choices = choices,
            value = value,
            tooltip = "Which protocol to use to extract the ROI."
        )
        self.changed.connect(self.update_method_callback)
        self.events = EmitterGroup(
            source=self,
        )
        self.events.add(
            updated_method=Event,
        )

    def update_method_callback(self, method_name : str):
        protocol = self.current_protocol
        self.tooltip = getdoc(protocol.extract)
        self.events.updated_method()

    @property
    def current_protocol(self):
        return next(
            x for x in self.current_region.protocols
            if x.name == self.value
        ) 

    def on_update_region(self, region_name : str):
        region = next(x for x in REGIONS if region_name in x.alias_list)
        self.current_region = region
        self.choices = list(fn.name for fn in region.protocols)
        self.value = region.default_fcn_str