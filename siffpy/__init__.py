from __future__ import annotations

from siffpy.core import SiffReader, fit_exp, __version__ as _version

#TODO: IMPLEMENT SIFFTOTIFF import siffreader

__version__ = _version

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


    


        

