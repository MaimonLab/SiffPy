import traceback

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from magicgui import widgets
from napari.layers import Shapes
from napari.utils.events import EmitterGroup, Event
from PyQt5.QtWidgets import QMessageBox
from scipy.ndimage import convolve

from siffpy import SiffReader
from siffpy.siffplot.roi_protocols.protocerebral_bridge.von_mises.numpy_implementation import (
    VonMisesCollection, match_to_von_mises
)

mpl.use('Qt5Agg')
fdict = {
    'font.family':  'Arial',
    'font.size':    4.0,
    'text.color' : '#FFFFFF',
    'figure.dpi' : 300,
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 1.0, 0.0, 0.0),
    "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
    "xtick.color" : "#FFFFFF",
    "ytick.color" : "#FFFFFF",
    "axes.edgecolor" : "#FFFFFF",
    "axes.labelcolor" : "#FFFFFF",
    "legend.frameon" : False,
    "legend.handlelength" : 0,
    "legend.handletextpad" : 0,
}

mpl.rcParams.update(**fdict)

def warning_window(func):
    def wrapper(*args, **kwargs):
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


class CorrelationMatrices():
    
    def __init__(self):
        fig = Figure(figsize=(2,3), dpi=160)
        fig.patch.set_facecolor('#000000')
        fig.suptitle('Correlation matrices', color='#FFFFFF')
        source_axes = fig.add_subplot(1,2,1)
        von_mises_axes = fig.add_subplot(1,2,2)

        source_axes.set_title(
            'Source data', color = '#FFFFFF', fontdict={'fontsize': 3}
        )

        von_mises_axes.set_title(
            'von Mises fit', color = '#FFFFFF', fontdict={'fontsize': 3}
        )

        def format_corrmap(ax: Axes):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        format_corrmap(source_axes)
        format_corrmap(von_mises_axes)

        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setMaximumHeight(250)
        canvas.setMaximumWidth(400)

        self.source_axes = source_axes
        self.von_mises_axes = von_mises_axes
        self.canvas = canvas

        ### EVENTS

        self.events = EmitterGroup(
            source=self,
        )

        self.events.add(
            source_image_produced = Event,
            correlation_matrix_computed = Event,
        )

        ## WIDGETS

        self.slider = widgets.FloatSlider(
            label = 'kappa',
            orientation='horizontal',
            value=2.0,
            min=0.0,
            max=10.0,
        )

        self.lower_bound_slider = widgets.Slider(
            label = 'Correlation\nstart',
            min = 0,
            value = 1000,
        )

        self.upper_bound_slider = widgets.Slider(
            label = 'Correlation\nend',
            min = 0,
            value = 2000,
        )

        self.get_correlation_button = widgets.PushButton(
            text = 'Correlate drawn ROIs',
            tooltip = "Uses the `ROI shapes` layer from the source `ROIViewer`"
        )

        self.correlate_whole_image_button = widgets.PushButton(
            text = 'Correlate image with seeds',
            tooltip = "Correlates all pixels in image"
        )

        ## CALLBACKS

        self.slider.changed.connect(self.on_kappa_slider_change)
        
        self.get_correlation_button.clicked.connect(
            self.correlate_seeds
        )

        self.correlate_whole_image_button.clicked.connect(
            self.correlate_all_pixels
        )

        self.params_container = widgets.Container(
            layout = 'Horizontal',
            widgets = [
                self.slider,
                self.lower_bound_slider,
                self.upper_bound_slider,
                self.get_correlation_button,
                self.correlate_whole_image_button,
            ]
        )
        self.params_container.max_height = 200
        self.params_container.max_width = 400

    def update_source(self, source_data):
        """ Better to set ydata and xdata than to replot but lazy """
        self.source_data = source_data
        self.source_axes.imshow(source_data)
        self.on_kappa_slider_change(self.slider.value)
        self.canvas.draw()
    
    def update_von_mises(self, von_mises_data):
        self.von_mises_data = von_mises_data
        self.von_mises_axes.imshow(von_mises_data)
        self.canvas.draw()

    def on_kappa_slider_change(self, kappa : float):
        self.fits = VonMisesCollection(
            match_to_von_mises(self.source_data, kappa)
        )

        self.update_von_mises(self.fits.cov)

    def link_siffreader(self, siffreader : SiffReader):
        self.siffreader = siffreader
        self.lower_bound_slider.max = siffreader.im_params.num_timepoints
        self.upper_bound_slider.max = siffreader.im_params.num_timepoints

    def provide_roi_shapes_layer(self, layer : Shapes, image_shape : tuple):
        self.source_rois_layer = layer
        self.image_shape = image_shape
    
    def correlate_seeds(self):
        """ Called when correlating the drawn ROIs to estimate a phase"""
        try:
            if not hasattr(self, 'siffreader'):
                raise ValueError("No siffreader provided")
            if not hasattr(self, 'source_rois_layer'):
                raise ValueError("No source ROI layer provided")
            frames = self.siffreader.get_frames(
                self.siffreader.im_params.flatten_by_timepoints(
                    int(self.lower_bound_slider.value),
                    int(self.upper_bound_slider.value),
                ),
                self.siffreader.registration_dict,
            ).reshape((-1, *self.siffreader.im_params.volume))

            self.seed_t_series = np.array([
                    frames[:,mask].sum(axis=1)
                    for mask in self.correlation_rois.reshape(
                        (-1, *self.siffreader.im_params.volume)
                    )
                    ]
                ) # nROIs by nTimepoints
            
            self.update_source(np.corrcoef(self.seed_t_series))
            self.events.correlation_matrix_computed()

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error: {e}")
            msg.setInformativeText(
                f"Traceback: {traceback.format_exc()}"
            )
            msg.setWindowTitle("Error")
            msg.exec_()
            
    def correlate_all_pixels(self):
        """
        Correlate each pixel with the drawn ROIs,
        take a vector sum of the pixel correlations with phase,
        and store that complex valued array (which will produce
        the source image in the main window).
        """
        try:
            if not hasattr(self, 'fits'):
                raise ValueError("No fits provided")
            
            frames = self.siffreader.get_frames(
                self.siffreader.im_params.flatten_by_timepoints(
                    int(self.lower_bound_slider.value),
                    int(self.upper_bound_slider.value),
                ),
                self.siffreader.registration_dict,
            ).reshape((-1, *self.siffreader.im_params.volume))
            frames = frames.reshape((frames.shape[0], -1)) # timepoints by pixels

            correlation = (
                self.seed_t_series @ frames/frames.shape[0] - #E[XY]
                np.outer(
                    self.seed_t_series.mean(axis=1), #E[X]
                    frames.mean(axis=0) #E[Y]
                )
            ) / np.outer(
                self.seed_t_series.std(axis=1), #std(X)
                frames.std(axis=0) #std(Y)
            )

            self.fft_image = (
                np.exp(1j * self.fits.means).T @
                correlation/len(self.fits)
            ).reshape(self.siffreader.im_params.volume)

            weights = (1, 1, *[x//50 for x in self.siffreader.im_params.shape])

            self.fft_image = convolve(
                self.fft_image,
                np.ones(weights)/np.prod(weights),
                mode='wrap',
            )

            self.events.source_image_produced()

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error correlating image: {e}")
            msg.setInformativeText(
                f"Traceback: {traceback.format_exc()}"
            )
            msg.setWindowTitle("Error")
            msg.exec_()

    @property
    def correlation_rois(self) -> np.ndarray:
        if not hasattr(self, 'source_rois_layer'):
            raise ValueError("No source ROI layer provided")
        return self.source_rois_layer.to_masks(self.image_shape)

    @property
    def widget(self):
        return self.params_container