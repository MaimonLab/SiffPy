from siffpy.sifftrac.plotters.tracplotter import TracPlotter
from .log_interpreter.fictraclog import *
from .plotters import *

def load_plotters(path : str = None)->list[TracPlotter]:
    """
    Finds saved plotters in path and returns them in a list
    """
    raise NotImplementedError()