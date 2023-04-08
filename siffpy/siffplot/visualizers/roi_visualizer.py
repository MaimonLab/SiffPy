from siffpy.siffplot.siffvisualizer import SiffVisualizer
from siffpy.siffplot.napari_viewers import ROIViewer
from siffpy.siffplot import roi_protocols

from siffpy.core import SiffReader

class ROIVisualizer(SiffVisualizer):
    """
    Extends the SiffVisualizer to provide
    annotation of reference frames to select
    ROIs.
    """

    ROI_ANNOTATION_LAYER_NAME = 'ROI shapes'

    def __init__(self, siffreader : SiffReader):
        super().__init__(siffreader)
        self.image_opts['clim'] = (0,1) # defaults to highest contrast

    def draw_rois(self, **kwargs)->None:
        """
        Returns a napari Viewer object that shows
        the reference frames of the .siff file and
        a layer for overlaying drawn polygons and shapes.
        """
        if not hasattr(self.siffreader, 'reference_frames'):
            raise AssertionError("SiffReader has no registered reference frames.")

        self.viewer = ROIViewer(self.siffreader, visualizer = self, title='Annotate ROIs')
        self.viewer.viewer.layers.selection = [self.viewer.viewer.layers['ROI shapes']] # selects the ROI drawing layer
        self.viewer.save_rois_fcn = self.save_rois

    def redraw_rois(self):
        """ Redraws the rois, for example after segmentation. """
        raise NotImplementedError()