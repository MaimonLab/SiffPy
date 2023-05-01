from __future__ import annotations

try:
    from siffpy.core._version import __version__, __version_tuple__
except ImportError:
    from siffpy.core.utils.shame import __version__, __version_tuple__
    print("Used shame.py. Please shame Stephen for not fixing this bug.")

from siffpy.core import SiffReader

#TODO: IMPLEMENT SIFFTOTIFF import siffreader

def siff_to_tiff(source_file : str, target_file : str = None)->None:
    """
    Converts a .siff file to a .tiff file containing only intensity information. For siffcompressed
    data, this should be smaller (because it just discards arrival times). For siff data that's not too big,
    this will likely create a larger file.

    INPUTS
    ------

    source_file : str

        Path to a .siff file

    target_file : str

        Path to where the .tiff should be saved
    """
    raise NotImplementedError()
    siffreader.siff_to_tiff(source_file, savepath=target_file)


    


        

