"""
Base class of the NapariInterface, a family of
classes that do not subclass the napari.Viewer
object, but mostly behave like they do.

SCT Dec 28, 2021
"""
import traceback

import napari
from napari.layers import Shapes
from PyQt5.QtWidgets import QMessageBox

from ...core import SiffReader
from ..roi_protocols.utils.napari_fcns import rois_into_shapes_layer

ROI_OBJECT_LAYER_NAME = "Primary ROIs"

class NapariInterface():
    """
    All NapariInterfaces have an attribute
    'viewer' which is a napari.Viewer object that
    can be accessed by treating the NapariInterface
    like a viewer itself.
    """

    def __init__(self, siffreader : SiffReader, *args, visualizer = None, **kwargs):
        """
        Accepts all napari.Viewer arguments plus requires a siffpy.SiffReader
        object as its first argument

        Arguments
        ---------

        siffreader : SiffReader

            A SiffReader object for accessing file information

        visualizer : SiffVisualizer (optional)

            The SiffVisualizer object linked to this NapariInterface so that the NapariInterface
            can sometimes call SiffVisualizer functions

            Can be left blank
        """
        self.viewer : napari.Viewer = napari.Viewer(*args, **kwargs)
        self.siffreader : SiffReader = siffreader
        self.scale = self.siffreader.im_params.scale[1:] #ignore the time dimension
        self.visualizer = visualizer # I don't like this style but it ends up being easier here for them to point to one another.

    def add_roi_object_layer(self, visible : bool = False):
        """ Creates an roi_object_layer that might be broadly useful """
        self.viewer.add_shapes(
                name = ROI_OBJECT_LAYER_NAME,
                ndim = 3,
                scale = self.scale,
                visible = visible,
        )

    @property
    def roi_object_layer(self) -> Shapes:
        """ May not exist for all viewers or even may be deleted for some reason. """
        roi_object_layer = next(
            filter(
                    lambda x: x.name == ROI_OBJECT_LAYER_NAME,
                    self.viewer.layers
                ),
            None
        )
        return roi_object_layer

    def draw_rois_on_napari(self):
        """ Draws all stored rois in the visualizer on the napari Viewer object"""
        roi_layer = self.roi_object_layer
        if roi_layer is None:
            raise AssertionError("No ROI layer created for this NapariInterface.")
        
        roi_layer.visible = False

        if self.visualizer is None:
            return
        if not hasattr(self.visualizer, 'rois'):
            return
        if self.visualizer.rois is None:
            return
        if len(self.visualizer.rois) == 0:
            return

        # iterate through each, draw it on the roi_layer.
        rois_into_shapes_layer(self.visualizer.rois, roi_layer)
        roi_layer.visible = True

    def __getattr__(self, attr: str):
        """
        If you try to get an attribute but it's not
        an attribute of the NapariInterface itself, try seeing if
        it's an attribute of its Viewer
        """
        try:
            return getattr(self.viewer, attr)
        except AttributeError:
            raise AttributeError(f"Requested attribute {attr} is not an attribute or method of this {self.__class__.__name__} nor of a napari.Viewer object.")

    def warning_window(self, warning_msg : str, exception : Exception = None):
        """ Opens a new window with a warning message. """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(warning_msg)
        msg.setWindowTitle(f"Warning: ({self.viewer.title})")
        if isinstance(exception, Exception):
            traceback_str = "\n".join(traceback.TracebackException.from_exception(exception).format())
            msg.setDetailedText(traceback_str)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
