# Functions for circularizing floats and ints

import numpy as np

def zeroed_circ(x : float, rollover : float) -> float:
    """ Returns a circularized variable that goes from -rollover/2 to +rollover/2"""
    return (x + rollover/2) % rollover - rollover/2

def circ_d(x : float, y : float, rollover : float)->float:
    """Wrapped-around distance between x and y"""
    return ((x-y + rollover/2) % rollover) - rollover/2

def re_circ(x : float, rollover : float) -> float:
    """ Takes de-circularized data and reverts it to circularized """
    return (x + rollover) % rollover

def roll_d(roll1 : tuple[float, float], roll2: tuple[float,float], rollover_y: float, rollover_x : float)->float:
    """ Distance between two rollovers """
    d_y = circ_d(roll1[0],roll2[0],rollover_y)
    d_x = circ_d(roll1[1],roll2[1],rollover_x)
    return np.sqrt(d_x**2 + d_y**2)