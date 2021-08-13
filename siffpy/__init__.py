from __future__ import annotations
from .siffpy import *
#from . import siffplot
from . import siffutils

def siff_to_tiff(filename : str)->None:
    """
    Converts a .siff file to a .tiff file containing only intensity information. For siffcompressed
    data, this should be smaller (because it just discards arrival times). For siff data that's not too big,
    this will likely create a larger file.

    INPUTS
    ------

    filename (str):

        Path to a .siff file
    """
    import siffreader
    raise NotImplementedError("Siff_to_tiff not yet implemented.")
