from __future__ import annotations
from typing import Optional

try:
    from siffpy.core._version import __version__, __version_tuple__
except ImportError:
    from siffpy.core.utils.shame import __version__, __version_tuple__
    print("Used shame.py. Please shame Stephen for not fixing this bug.")

from siffpy.core import SiffReader
from siffpy.core import ImParams
from siffpy.core import FLIMParams
from siffpy.siffmath import FlimTrace

def siff_to_tiff(
        source_file : str,
        target_file : Optional[str] = None,
        mode : str = 'scanimage'
    )->None:
    """
    Converts a .siff file to a .tiff file containing only intensity information. For siffcompressed
    data, this should be smaller (because it just discards arrival times). For siff data that's not too big,
    this will likely create a larger file.

    Arguments
    ------

    source_file : str
        Path to a .siff file

    target_file : Optional[str]
        Path to where the .tiff should be saved. If `None`, saves
        to the same directory as the source file, with the same name
        but with the extension changed to .tiff

    mode : str
        Can be `scanimage` or (in progress) `ome`. `scanimage` is the default.
        This is not OME-compliant, so ImageJ/Fiji (for example) will just parse
        it as a flattened image series. When `ome` is implemented, that mode will
        be readable by most bio image viewers.
    """
    import siffreadermodule
    siffreadermodule.siff_to_tiff(source_file, savepath=target_file, mode=mode)

def siff_to_tiff_command_line(argv):
    """ Function used as an entry point from the command line """
    raise NotImplementedError("Command line calls of siff-to-tiff are not yet supported!")
    pass
    


        

