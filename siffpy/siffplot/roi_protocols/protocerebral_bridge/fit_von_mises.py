# Code for ROI extraction from the protocerebral bridge after manual input

# Feels like I use too much interfacing with the CorrelationWindow class,
# which seems like it should do as much as possible without interacting with
# this stuff...

import numpy as np

from siffpy import SiffReader
from siffpy.siffplot.roi_protocols import rois, ROIProtocol
from siffpy.siffplot.roi_protocols.utils import PolygonSource
from siffpy.siffplot.roi_protocols.protocerebral_bridge.von_mises.napari_tools import (
    CorrelationWindow
)

class FitVonMises(ROIProtocol):

    name = "Fit von Mises"
    base_roi_text = "View correlation map"

    def on_click(self, extraction_initiated_event):
        """
        Opens a correlation window to allow annotation
        of the protocerebral bridge, passes the
        "extraction initiated" event to the correlation
        window's done button.
        """
        viewer = extraction_initiated_event.source.napari_interface
        corr_window = CorrelationWindow(viewer)
        corr_window.done_button.clicked.connect(
            lambda *args: extraction_initiated_event()
        )
        sr : SiffReader = viewer.siffreader
        corr_window.provide_roi_shapes_layer(
            viewer.drawn_rois_layer,
            sr.im_params.single_channel_volume,
        )
        corr_window.link_siffreader(sr)
        self.corr_window = corr_window
        
    def extract(
            self,
            reference_frames : np.ndarray,
            polygon_source : PolygonSource,
    )->rois.GlobularMustache:
        """
        Returns a GlobularMustache ROI made up of the individual
        masks extracted by correlating every pixel to the source ROIs
        """
        if not hasattr(self, 'corr_window'):
            raise RuntimeError("Correlation window not initialized -- run protocol first!")
        
        return rois.GlobularMustache(
            globular_glomeruli_masks = self.corr_window.seed_manager.get_masks()
        )
