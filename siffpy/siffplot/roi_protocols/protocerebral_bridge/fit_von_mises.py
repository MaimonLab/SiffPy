# Code for ROI extraction from the protocerebral bridge after manual input
import numpy as np
from holoviews import Polygons
from scipy.special import i0

from siffpy import SiffReader
from siffpy.siffplot.roi_protocols import rois, ROIProtocol
from siffpy.siffplot.roi_protocols.utils import (
    PolygonSource, polygon_to_mask, polygon_to_z
)
from siffpy.siffplot.roi_protocols.protocerebral_bridge.von_mises.numpy_implementation import (
    match_to_von_mises
)
from siffpy.siffplot.roi_protocols.protocerebral_bridge.von_mises.napari_tools import (
    CorrelationWindow
)

def correlate_seeds(
        siffreader : SiffReader,
        seed_rois : np.ndarray,
        correlation_window : CorrelationWindow,
    ):
    timepoint_bounds = (
        int(correlation_window.corr_mat_widget.lower_bound_slider.value),
        int(correlation_window.corr_mat_widget.upper_bound_slider.value),
    )

#    siffreader.sum_roi(
#
#    )

#    correlation_window.set_source_correlation(
#        corr_mat
#    )

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
        corr_window = CorrelationWindow()
        corr_window.done_button.clicked.connect(
            lambda *args: extraction_initiated_event()
        )
        sr : SiffReader = extraction_initiated_event.source.napari_interface.siffreader
        corr_window.corr_mat_widget.upper_bound_slider.max = sr.im_params.num_timepoints
        
        corr_window.corr_mat_widget.provide_roi_shapes_layer(
            extraction_initiated_event.source.napari_interface.roi_shapes_layer
        )
        corr_window.corr_mat_widget.set_correlation_button_callback(
            lambda *args: correlate_seeds(sr, corr_window.corr_mat_widget.correlation_rois, corr_window)
        )



    def extract(
            self,
            reference_frames : np.ndarray,
            polygon_source : PolygonSource,
    )->rois.GlobularMustache:
        raise NotImplementedError("Sorry bub!")