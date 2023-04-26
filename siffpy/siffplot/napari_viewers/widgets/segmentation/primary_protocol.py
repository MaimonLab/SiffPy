import inspect

from magicgui import widgets
from napari.utils.events import EventEmitter

from siffpy.siffroi.roi_protocols import REGIONS
from siffpy.siffplot.napari_viewers.widgets.segmentation.primary import (
    AnatomicalRegionBox, ExtractionMethodBox, ExtractionParamsContainer
)

class PrimaryProtocol():

    def __init__(self):

        ## Create the widgets
        self.anatomical_region_box = AnatomicalRegionBox(
            regions = REGIONS
        )

        self.extraction_method_box = ExtractionMethodBox(
            choices = list(protocol.name for protocol in self.current_region.protocols),
            value = self.current_region.default_fcn_str
        )
        self.extraction_method_box.current_region = self.current_region

        self.roi_class = widgets.Label(
            name = "ROIClass",
            label = "ROI Class : ",
            value = self.current_region.default_protocol.return_class.__name__,
            tooltip = inspect.getdoc(self.current_region.default_protocol.return_class)
        )

        self.extraction_params_container = ExtractionParamsContainer(
            initial_protocol = self.current_region.default_protocol
        )

        ## Connect them all to one another
        self.anatomical_region_box.changed.connect(
            self.extraction_method_box.on_update_region
        )

        self.updated_method.connect(
            self.extraction_params_container.on_update_method
        )

        self.updated_method.connect(
            self.update_roi_class
        )

        self.widgets = [
            self.anatomical_region_box,
            self.extraction_method_box,
            self.roi_class,
            self.extraction_params_container
        ]

    @property
    def current_region(self):
        return self.anatomical_region_box.current_region

    @property
    def current_protocol(self):
        return self.extraction_params_container.current_protocol

    @property
    def extraction_button_clicked(self)->EventEmitter:
        return self.extraction_params_container.events.extraction_button_clicked
    
    @property
    def updated_method(self)->EventEmitter:
        return self.extraction_method_box.events.updated_method

    @property
    def extraction_kwargs(self)->dict:
        return self.extraction_params_container.to_dict()

    def update_roi_class(self, event):
        returned_class = event.source.current_protocol.return_class
        try:
            self.roi_class.tooltip = inspect.getdoc(returned_class)
        except:
            self.roi_class.tooltip = "No docstring found"
        try:
            self.roi_class.value = returned_class.__name__
        except:
            self.roi_class.value = "Invalid function"