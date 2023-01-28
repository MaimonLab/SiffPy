import magicgui.widgets as widgets

from siffpy.siffplot.napari_viewers.napari_interface import NapariInterface
from siffpy.siffplot.utils.exceptions import NoROIException
from siffpy.siffplot.roi_protocols import ROI

class ROIsContainer(widgets.Container):
    def __init__(self, window : NapariInterface):
        super().__init__(
            name = "CurrentROIsContainer",
            layout = 'horizontal',
            label = "Current ROIs: ",
            tooltip = "ROIs stored in the SiffVisualizer's ROIs attribute.",
            visible = False
        )

        self.window = window

    def refresh(self):
        try:
            self.window.draw_rois_on_napari()
        except NoROIException:
            return
        
        self.clear()
        self.visible = False
        if self.window.visualizer is None:
            return
        if not hasattr(self.window.visualizer, 'rois'):
            return
        if self.window.visualizer.rois is None:
            return
        if len(self.window.visualizer.rois) == 0:
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

        self.extend([names, classes, segmenteds, nsegs])

        roi : ROI
        for roi in self.window.visualizer.rois:
            names.append(widgets.Label(label=roi.name))
            classes.append(widgets.Label(label=type(roi).__name__))
            if hasattr(roi,'subROIs'):
                segmenteds.append(widgets.Label(label="Y"))
                nsegs.append(widgets.Label(label=str(len(roi.subROIs))))
            else:
                segmenteds.append(widgets.Label(label="N"))
                nsegs.append(widgets.Label(label="N/A"))

        self.visible = True

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