"""
Base class of the NapariInterface, a family of
classes that do not subclass the napari.Viewer
object, but mostly behave like they do.

SCT Dec 28, 2021
"""
import napari

from ...siffpy import SiffReader

class NapariInterface():
    """
    All NapariInterfaces have an attribute
    'viewer' which is a napari.Viewer object that
    can be accessed by treating the NapariInterface
    like a viewer itself.
    """

    def __init__(self, siffreader : SiffReader, *args, **kwargs):
        """
        Accepts all napari.Viewer arguments plus requires a siffpy.SiffReader
        object as its first argument
        """
        self.viewer : napari.Viewer = napari.Viewer(*args, **kwargs)
        self.siffreader : SiffReader = siffreader

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