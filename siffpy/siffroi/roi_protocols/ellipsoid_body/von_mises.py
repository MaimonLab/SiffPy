"""
Implements an identical protocol to the
Protocerebral Bridge one but returns an Ellipse
"""

# Code for ROI extraction from the protocerebral bridge after manual input

# Feels like I use too much interfacing with the CorrelationWindow class,
# which seems like it should do as much as possible without interacting with
# this stuff...

import numpy as np

from siffpy.siffroi.roi_protocols import rois
from siffpy.siffroi.roi_protocols.utils import PolygonSource
from siffpy.siffroi.roi_protocols.protocerebral_bridge.fit_von_mises import FitVonMises

class FitVonMisesEB(FitVonMises):

    name = "Fit von Mises"
    base_roi_text = "View correlation map"
        
    def extract(
            self,
            reference_frames : np.ndarray,
            polygon_source : PolygonSource,
    )->rois.Ellipse:
        raise NotImplementedError("Sorry bub!")