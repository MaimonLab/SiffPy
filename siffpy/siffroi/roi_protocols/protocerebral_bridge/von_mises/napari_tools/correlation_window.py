from functools import wraps
import traceback

import numpy as np
from magicgui import widgets
import napari
from napari.layers import Shapes
from qtpy.QtWidgets import QMessageBox

from siffpy import SiffReader
from siffpy.siffroi.roi_protocols.protocerebral_bridge.von_mises.napari_tools.correlation_matrix_widget import (
    CorrelationMatrices
)
from siffpy.siffroi.roi_protocols.protocerebral_bridge.von_mises.napari_tools.flower_plot import (
    FlowerPlot
)
from siffpy.siffroi.roi_protocols.protocerebral_bridge.von_mises.napari_tools.seeds import (
    SeedManager
)

from matplotlib.pyplot import get_cmap

hsv = get_cmap('hsv')

def fft_to_rgba(fft, cmap = hsv) -> np.ndarray:
    """
    Converts a complex fft to an rgba image
    using the phase for the color and
    the amplitude for the alpha
    """
    rgba = cmap(
        (np.angle(fft) + np.pi)/(2*np.pi),
        alpha = np.abs(fft)/np.nanmax(np.abs(fft)),
    )
    return rgba

def warning_window(func):
    """
    Produces a warning message box if an
    error occurs in the wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Try to execute func, if there's a failure
        spit out the error message using the GUI
        (with a traceback!)
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"An error occurred: {e}")
            msg.setInformativeText(
                f"Traceback: {traceback.format_exc()}"
            )
            msg.setWindowTitle("Error")
            msg.exec_()
    return wrapper


class CorrelationWindow():
    """
    Does not interact with siffreader or protocols
    except to get the source image(s) and when
    the user clicks done. Both are handled by events
    outside of the `CorrelationWindow` class.
    """
    def __init__(self, source_image : np.ndarray = None):
        
        self.viewer = napari.Viewer(
            title = "Von Mises Correlation Analysis"
        )

        # Clean up the left side of the window
        self.viewer.window._qt_window._qt_viewer.dockLayerControls.setMaximumHeight(200)

        self.viewer.window._qt_window._qt_viewer.dockLayerList.setMaximumHeight(400)

        # Sets up the correlation matrices
        self.corr_mat_widget = CorrelationMatrices()

        self.viewer.window.add_dock_widget(
            self.corr_mat_widget.canvas, name = 'Correlation matrices',
            area = 'left',
        )

        self.viewer.window.add_dock_widget(
            self.corr_mat_widget.widget, name = 'Fit parameters',
            area = 'left',
        )

        self.flower_plot = FlowerPlot()
        self.viewer.window.add_dock_widget(
            self.flower_plot.canvas, name = 'Seed tunings',
        )

        done_button = widgets.PushButton(
            text = 'Done!'
        )
        self.done_button = done_button

        self.viewer.window.add_dock_widget(
            self.done_button,
        )

        self.seed_manager = SeedManager(
            self.flower_plot.figure,
            self.flower_plot.axes,
            source_image = source_image,
        )

        if isinstance(source_image, np.ndarray) and (
            source_image.dtype == np.complex64 or 
            source_image.dtype == np.complex128
        ):
            print("Setting source images")
            self.set_source_image(source_image)

        self.seed_manager.events.changed.connect(
            self.update_masks
        )

        self.corr_mat_widget.events.source_image_produced.connect(
            lambda x: self.set_source_image(x.source.fft_image)
        )

        self.last_selected_points = set()

    @warning_window
    def set_source_image(
            self,
            source_image : np.ndarray,
            cmap = hsv, 
        ):
        """
        Takes a complex-valued source image and converts it
        to RGBA for display in napari.
        """
        source_image = source_image.squeeze()
        if source_image.ndim != 3:
            raise ValueError(
                f"Source image must have exactly 3 non-singleton dimensions. Tried to pass {source_image.ndim}"
            )

        if 'Seed correlation map' in self.viewer.layers:
            self.viewer.layers['Seed correlation map'].data = fft_to_rgba(source_image, cmap)

        else:
            self.viewer.add_image(
                fft_to_rgba(source_image, cmap),
                name = 'Seed correlation map',
                rgb = True,
            )

            layer_scale = self.viewer.layers['Seed correlation map'].scale

            self.viewer.add_image(
                np.zeros_like(source_image, dtype = float),
                name = 'Seed masks',
                channel_axis = None,
                blending = 'additive',
                opacity = 0.18,
                rgb = False,
                scale= layer_scale,
            )

            self.viewer.add_image(
                np.zeros_like(source_image, dtype = float),
                name = 'Seed masks (selected)',
                channel_axis = None,
                blending = 'additive',
                opacity = 0.5,
                rgb = False,
                scale= layer_scale,
            )

            self.viewer.add_points(
                name = 'Seed points',
                size = 6,
                face_color = '#FFFFFF',
                edge_color= '#000000',
                edge_width = 0.3,
                opacity=1.0,
                out_of_slice_display=True,
                scale= layer_scale,
            )

            self.viewer.layers['Seed points'].events.data.connect(
                self.on_seed_pixel_change,
                'on_seed_pixel_change'
            )

            self.viewer.layers['Seed points'].events.highlight.connect(
                self.on_selected_seed,
                'on_selected_seed',
            )

            self.selected_data = set()

        self.seed_manager.set_source_image(source_image)

    def on_seed_pixel_change(self, event):
        px : np.ndarray # index of the seed pixel
        for px in event.source.data:
            if not (px in self.seed_manager):
                nearest_pixel = tuple(px.astype(int))
                rgb_value = self.viewer.layers['Seed correlation map'].data[nearest_pixel][:-1] # remove the alpha
                fft_value = self.seed_manager.source_image[nearest_pixel]
                self.seed_manager.create_seed(
                   px,
                   rgb_value,
                   fft_value,
                )

        # get rid of lingering seeds
        for seed in self.seed_manager:
            if not any(
                np.array_equal(seed, point)
                for point in event.source.data.tolist()
            ):
                self.seed_manager.remove_seed(seed)

        self.seed_manager.events.changed() 
        
    @warning_window
    def on_selected_seed(self, event):
        if self.last_selected_points == event.source.selected_data:
            return
        
        for pt_idx in event.source.selected_data:
            pt_loc = event.source.data[pt_idx]
            if pt_loc in self.seed_manager:
                self.seed_manager[pt_loc].on_px_select()
        
        for pt_idx in range(len(event.source.data)):
            if not pt_idx in event.source.selected_data:
                pt_loc = event.source.data[pt_idx]
                if pt_loc in self.seed_manager:
                    self.seed_manager[pt_loc].on_px_deselect()

        self.last_selected_points = event.source.selected_data

    def link_siffreader(self, siffreader : SiffReader):
        self.corr_mat_widget.link_siffreader(siffreader)

    def update_masks(self, event):
        """ Called by the seed manager when the mask changes"""
        self.viewer.layers['Seed masks'].data = self.seed_manager.mask
        self.viewer.layers['Seed masks (selected)'].data = self.seed_manager.selected_mask

    def set_source_correlation(self, source_corr : np.ndarray):
        self.corr_mat_widget.update_source(source_corr)

    def provide_roi_shapes_layer(self, shapes_layer : Shapes, image_shape : tuple,):
        self.corr_mat_widget.provide_roi_shapes_layer(shapes_layer, image_shape)
