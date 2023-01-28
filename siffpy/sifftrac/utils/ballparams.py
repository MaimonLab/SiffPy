# parameters describing a fictrac ball
from dataclasses import dataclass
DEFAULT_PARAMS = {
    'radius'   : 3,
    'units'    : 'mm',
    'axis'     : 'free',
    'material' : 'foam'
}

@dataclass
class BallParams():
    radius : float = 3
    units : str = 'mm'
    axis : str = 'free'
    material : str = 'foam'