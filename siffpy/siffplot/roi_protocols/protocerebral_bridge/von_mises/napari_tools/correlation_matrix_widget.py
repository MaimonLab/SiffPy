from typing import Callable

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
            text = 'Get correlations with source ROIs',
            tooltip = "Uses the `ROI shapes` layer from the source `ROIViewer`"
        )

        self.slider.changed.connect(self.on_kappa_slider_change)

        self.params_container = widgets.Container(
            layout = 'Horizontal',
            widgets = [
                self.slider,
                self.lower_bound_slider,
                self.upper_bound_slider,
                self.get_correlation_button
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

    def provide_roi_shapes_layer(self, layer : Shapes):
        self.source_rois_layer = layer

    def set_correlation_button_callback(self, callback : Callable):
        self.get_correlation_button.changed.connect(callback)

    @property
    def correlation_rois(self) -> np.ndarray:
        if not hasattr(self, 'source_rois_layer'):
            raise ValueError("No source ROI layer provided")
        return self.source_rois_layer.to_masks()

    @property
    def correlation_button_clicked(self):
        return self.get_correlation_button.clicked

    @property
    def widget(self):
        return self.params_container