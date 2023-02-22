from enum import Enum
import typing, inspect
from typing import Callable
from inspect import Parameter

import numpy as np
import magicgui.widgets as widgets

from siffpy.siffplot.napari_viewers.napari_interface import NapariInterface
from siffpy.core import SiffReader
from siffpy.siffplot.utils.exceptions import NoROIException
from siffpy.siffplot.roi_protocols import REGIONS, roi_protocol
from siffpy.siffplot.roi_protocols.rois import ROI
from siffpy.siffplot.roi_protocols.utils import napari_fcns
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

        self.segmentation_fcn = segmentation_fcn
        self.save_rois_fcn : Callable = None

        self.initialize_layers(edge_color = edge_color)
        #roi_widget = self.initialize_segmentation_widget()
        roi_widget = SegmentationWidget(self, self.segmented_rois_layer)
        roi_widget.connect_reference_frames(self.siffreader.reference_frames)
        self.roi_widgets = roi_widget
        self.viewer.window.add_dock_widget(roi_widget, name='ROI segmentation tools')

    
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