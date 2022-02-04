HOLOVIEWS = False
try:
    import holoviews # testing if the import works
    from siffpy.sifftrac.plotters.tracplotter import TracPlotter
    from .plotters import *
    HOLOVIEWS = True
except ImportError: # fine if you can't import holoviews
    pass

from .log_interpreter.fictraclog import FictracLog

if HOLOVIEWS:
    def load_plotters(path : str = None)->list[TracPlotter]:
        """
        Finds saved plotters in path and returns them in a list
        """
        raise NotImplementedError()