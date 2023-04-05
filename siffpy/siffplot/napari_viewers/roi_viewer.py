from typing import Callable

import numpy as np
from napari.utils.events import Event, EventEmitter, EmitterGroup

from siffpy.siffplot.napari_viewers.napari_interface import NapariInterface
from siffpy.core import SiffReader
from siffpy.siffplot.utils.exceptions import NoROIException
from siffpy.siffplot.roi_protocols.roi_protocol import ROIProtocol
from siffpy.siffplot.roi_protocols.utils.napari_fcns import PolygonSourceNapari
from siffpy.siffplot.napari_viewers.widgets import SegmentationWidget

CINNABAR = '#db544b'
DRAWN_SHAPE_LAYER_NAME = "ROI shapes"
SUBROI_LAYER_NAME = "Segmented ROIs"
ANATOMY_SHAPE_LAYER_NAME = "Anatomy references"

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
        self.events = EmitterGroup(
            source=self,
        )
        self.events.add(
            extraction_complete = Event,
            segmentation_complete = Event,
        )

        self.segmentation_fcn = segmentation_fcn
        self.save_rois_fcn : Callable = None

        self.initialize_layers(edge_color = edge_color)
        
        roi_widget = SegmentationWidget(self, self.segmented_rois_layer)
        roi_widget.events.extraction_initiated.connect(self.extract_rois)

        self.roi_widget = roi_widget
        
        self.viewer.window.add_dock_widget(roi_widget, name='ROI segmentation tools')
        roi_widget.update_roi_list(self.visualizer.rois)

    
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

        self.viewer.add_shapes(
            face_color = "transparent",
            name=ANATOMY_SHAPE_LAYER_NAME,
            ndim = 3,
            edge_color = "#FFFFFF",
            scale = self.scale
        )

    def extract_rois(self, event):
        """ Extracts ROIs using the protocol specified by the segmentation widget """
        source : SegmentationWidget = event.source
        protocol : ROIProtocol = source.current_protocol
        args = []
        kwargs = {}
        if protocol.uses_reference_frames:
            args.append(self.siffreader.reference_frames)
        if protocol.uses_polygon_source:
            args.append(self.polygon_source)
        try:
            self.visualizer.rois.append(protocol.extract(*args, **source.extraction_kwargs))
        except Exception as e:
            self.warning_window(f"Error in segmentation function: {e}", exception = e)
            return

        self.roi_widget.update_roi_list(self.visualizer.rois)

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
    
    @property
    def polygon_source(self) -> PolygonSourceNapari:
        """ May not exist for all viewers or even may be deleted for some reason. """
        return PolygonSourceNapari(self.viewer)