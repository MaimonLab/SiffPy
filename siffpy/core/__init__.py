try:
    from siffpy.core._version import __version__
except ImportError:
    from siffpy.core.utils.shame import __version__
from siffpy.core.siffreader import SiffReader
from siffpy.core.flim import FLIMParams