import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from napari.utils.events import EmitterGroup, Event
from skimage import morphology

mpl.use('Qt5Agg')
fdict = {
    'font.family':  'Arial',
    'font.size':    6.0,
    'text.color' : '#FFFFFF',
    'figure.dpi' : 200,
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

class RosettePetal():
    """
    Handles the matplotlib interaction for a single petal
    """
    def __init__(
        self,
        fig : Figure,
        ax : Axes,
        mean : float,
        width : float,
        mag : float = 1.0,
        color : tuple = (0,0,0),
    ):
        self.fig = fig
        self.ax = ax
        self.mean = mean
        self.width = width
        self.mag = mag
        self.color = color
        self.t_axis = np.linspace(0,1,100)
        self.clicked_pt_idx = None
        self.plt_idx = None
        self.events = EmitterGroup(
            self,
        )
        self.events.add(
            changed = Event,
            deleted = Event,
        )

        self.plot_petal()

        self.cidrelease = self.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        
        self.pickevent = self.canvas.mpl_connect(
            'pick_event', self.on_pick)
        
    def plot_petal(self):
        self.curve_line = self.ax.plot(
           *(self.t_coord_to_polar(self.t_axis, self.mean, self.width, self.mag)),
           color = self.color,
        )

        self.clickable_points = self.ax.scatter(
           *(self.t_coord_to_polar([0.25,0.5,0.75], self.mean, self.width, self.mag)),
           color = self.color,
           picker = True,
        )
        self.update_rose()
        self.canvas.draw()

    def update_rose(self):
        theta, r = self.t_coord_to_polar(self.t_axis, self.mean, self.width, self.mag)
        self.curve_line[0].set_xdata(theta)
        self.curve_line[0].set_ydata(r)

        pts = self.t_coord_to_polar([0.25,0.5,0.75], self.mean, self.width,self.mag)
        self.clickable_points.set_offsets(np.array(pts).T)
        self.events.changed()

    def px_selected(self):
        """ Handles when the source pixel (from the FFT) is selected """
        self.clickable_points.set_picker(True)
        for line in self.curve_line:
            line.set(alpha=1.0)
        self.clickable_points.set_alpha(1.0)
        self.canvas.draw()
    
    def px_deselected(self):
        """ Handles when the source pixel (from the FFT) is deselected"""
        self.clickable_points.set_picker(False)
        for line in self.curve_line:
            line.set(alpha=0.3)
        self.clickable_points.set_alpha(0.3)
        self.canvas.draw()

    def on_pick(self, event):
        """ Handles when the flower petal is selected! """
        if event.artist != self.clickable_points: return
        if not self.clickable_points.get_picker(): return
        if hasattr(event.ind, '__len__'):
            ind = event.ind[0]
        else:
            ind = event.ind
        self.clicked_pt_idx = int(ind)

    def on_release(self, event):
        self.clicked_pt_idx = None
        self.update_rose()
        self.canvas.draw()

    def on_motion(self, event):
        """ During dragging of petal points """
        if self.clicked_pt_idx is None: return
        if event.inaxes != self.ax: return

        if (self.clicked_pt_idx % 2): # center point
            self.mean = event.xdata
            self.mag = event.ydata
        else: # edge points
            self.width = 2*np.abs(np.angle(np.exp(1j*event.xdata)/np.exp(1j*self.mean)))

        self.update_rose()
        self.canvas.draw()

    @classmethod
    def t_coord_to_polar(cls, t_coord : float, mean : float, width : float, mag : float = 1.0):
        """ Returns a tuple of polar coordinates (theta, r) given a t coordinate
        or iterable of t coordinates, a mean angle, and a width angle."""
        t_coord = np.array(t_coord)
        theta = mean + 2*(t_coord-0.5)*width
        pseudotheta = (np.pi*(t_coord-0.5)) # Rescales the theta axis to be 0-2pi
        r = mag*np.cos(pseudotheta) # Rescales the r axis to be 0-1
        return theta, r

    @property
    def canvas(self):
        return self.fig.canvas
    
    @property
    def widget(self):
        return FigureCanvas(self.fig)

    def __del__(self):
        try:
            self.events.deleted()
            self.canvas.mpl_disconnect(self.cidrelease)
            self.canvas.mpl_disconnect(self.cidmotion)
            self.canvas.mpl_disconnect(self.pickevent)
            self.curve_line[0].remove()
            self.clickable_points.remove()
            self.canvas.draw()
        except:
            pass

class FlowerPlot():
    
    def __init__(self):
        fig = Figure(figsize=(3,3), dpi=160)
        fig.patch.set_facecolor('#000000')
        fig.suptitle('Seed ROI tuning', color='#FFFFFF')
        flower_axes = fig.add_subplot(1,1,1, projection = 'polar')
        flower_axes.set_rticks([])
        flower_axes.set_xticks([])
        flower_axes.set_facecolor('#000000')
        flower_axes.spines['polar'].set_color('#FFFFFF')
        flower_axes.spines['polar'].set_linewidth(2)
        flower_axes.spines['end'].set_color('#FFFFFF')
        flower_axes.spines['end'].set_linewidth(2)
        flower_axes.spines['start'].set_color('#FFFFFF')
        flower_axes.spines['inner'].set_color('#FFFFFF')

        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(350)
        canvas.setMinimumWidth(350)
        canvas.setMaximumHeight(450)
        self.canvas = canvas
        self.figure = fig
        self.axes = flower_axes

    @property
    def widget(self):
        return self.canvas