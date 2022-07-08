HOLOVIEWS = False
import warnings
from .log_interpreter import FictracLog
try:
    import holoviews # testing if the import works
    from siffpy.sifftrac.plotters.tracplotter import TracPlotter
    from .plotters import *
    HOLOVIEWS = True
except ImportError as e: # fine if you can't import holoviews
    warnings.warn(e.msg)    
    pass


if HOLOVIEWS:
    def load_plotters(path : str = None)->list[TracPlotter]:
        """
        Finds saved plotters in path and returns them in a list
        """
        raise NotImplementedError()