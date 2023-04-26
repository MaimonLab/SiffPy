import magicgui.widgets as widgets
from siffpy.siffroi.roi_protocols import Region, REGIONS

class AnatomicalRegionBox(widgets.ComboBox):
    """
    A relatively boring class, just a dropdown menu
    of anatomical regions.
    """
    def __init__(self, regions : list[Region]):
        super().__init__(
            label='Anatomical region',
            name='AnatomicalRegionBox',
            choices = list((region.region_enum.value for region in regions)),
            tooltip = 'The anatomical region determines which segmentation protocols can be used'
        )

    @property
    def current_region(self)->Region:
        return next(
            region for region in REGIONS
            if region.region_enum.value == self.current_choice
        )