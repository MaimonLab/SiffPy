import numpy as np
from .napari_interface import NapariInterface
from ...siffpy import SiffReader

CINNABAR = '#db544b'

class ROIViewer(NapariInterface):
    """
    Access to a napari Viewer object specialized for annotating ROIs.
    Designed to behave LIKE a Viewer without subclassing the Viewer
    directly.

    TODO: FINISH IMPLEMENTING. Most important feature: incorporate
    the roi functions into widgets docked on the viewer! This might
    actually end up being quite slick, with dropdown menu for brain
    regions that determine a separate dropdown menu for currently-
    implemented fitting functions.
    """

    def __init__(self, siffreader : SiffReader, *args, **kwargs):
        """
        Accepts all napari.Viewer arguments plus requires a siffpy.SiffReader
        object as its first argument

        TODO: Annotate kwargs
        """
        super().__init__(siffreader, *args, **kwargs)
        self.viewer.dims.axis_labels = ['Z planes', 'x', 'y']
        self.scale = self.siffreader.im_params.scale[1:] #ignore the time dimension

        self.add_image(
            data = np.array(self.siffreader.reference_frames),
            name='Reference frames',
            scale = self.scale,
        )

        edge_color = CINNABAR
        if edge_color in kwargs:
            edge_color = kwargs['edge_color']

        self.add_shapes(
            face_color="transparent",
            name="ROI shapes",
            ndim=3,
            edge_color=edge_color,
            scale = self.scale
        )