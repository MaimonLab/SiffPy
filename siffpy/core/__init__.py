try:
    from siffpy.core._version import __version__
except ImportError:
    from siffpy.core.utils.shame import __version__  # noqa: F401
from siffpy.core.siffreader import SiffReader # noqa: F401
from siffpy.core.flim import FLIMParams # noqa: F401
from siffpy.core.utils import ImParams # noqa: F401