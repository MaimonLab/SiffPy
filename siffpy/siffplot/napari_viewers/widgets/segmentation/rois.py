import magicgui.widgets as widgets
from napari.utils.events import Event, EventEmitter, EmitterGroup


from siffpy.siffplot.napari_viewers.napari_interface import NapariInterface
from siffpy.siffplot.utils.exceptions import NoROIException
from siffpy.siffplot.roi_protocols import ROI

def roi_to_label(roi : ROI):
    return f"""
    {roi.name} : {roi.__class__.__name__}
    (slice {roi.slice_idx}) ({len(roi.subROIs)} subROIs)
    """

class ROIsContainer(widgets.Select):
    def __init__(self):
        super().__init__(
            name = "CurrentROIsContainer",
            label = "Current ROIs: ",
            tooltip = "ROIs stored in the SiffVisualizer's ROIs attribute.",
            visible = True,
            allow_multiple=True,
        )

        self.events = EmitterGroup(
            source=self,
        )

        self.events.add(
            roi_selected = Event,
        )
        
        self.roi_dict = {    
        }

    def update_roi_list(self, roi_list : list[ROI]):
        self.roi_dict = {
            roi_to_label(roi) : roi for roi in roi_list
        }
        self.choices = list(self.roi_dict.keys())

    @property
    def current_rois(self)->list[ROI]:
        return [self.roi_dict[roi_label] for roi_label in self.current_choice]


class ShowSubROIs(widgets.CheckBox):
    def __init__(self):
        super().__init__(
            name = "ShowSubROIsBox",
            label = "Show sub-ROIs",
            tooltip = "Toggles whether sub-ROI masks are shown on the Napari viewer"
        )

class SaveRoisPushbutton(widgets.PushButton):
    def __init__(self):
        super().__init__(
            name="SaveRoiPushButton",
            label = "Save ROIs",
            tooltip = "Saves current ROIs stored in SiffViewer object"
        )

    def save(self):
        pass