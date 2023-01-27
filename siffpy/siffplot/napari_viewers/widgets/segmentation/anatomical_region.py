import magicgui.widgets as widgets
from siffpy.siffplot.roi_protocols import Region

class AnatomicalRegionBox(widgets.ComboBox):
    def __init__(self, regions : list[Region]):
        super().__init__(
            label='Anatomical region',
            name='AnatomicalRegionBox',
            choices = list((region.region_enum.value for region in regions)),
            tooltip = 'The anatomical region determines which segmentation protocols can be used'
        )

    def connect_extraction_method_box(self, extraction_method_box):
        self.changed.connect(
            extraction_method_box.update_region
        )