# Functions for circularizing floats and ints
from itertools import tee

import numpy as np

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..., (sn-2, sn-1), (sn-1, sn)"
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

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


def split_angles_to_dict(x_var : np.ndarray, y_var : np.ndarray, xlabel : str ='x', ylabel : str ='y')->list[dict]:
    """
    Returns a list of dicts that can be passed to a Holoviews Path object.

    Splits circular variables up whenever there's a shift > pi in magnitude.

    ARGUMENTS
    ---------
    
    x_var : 
        
        The independent variable (usually the kdim) to pass to Holoviews

    y_var :

        The dependent variable (usually to be the vdim) to pass to Holoviews

    xlabel : str

        What the key for the x_var should be

    ylabel : str

        What the key for the y_var should be

    RETURNS
    -------

    split : list[dict]

        A list of dicts corresponding to individual line segments before they need to be
        made disjoint. Schematized as:

        [
            {
                xlabel : consecutive_x_coords,
                ylabel : consecutive_y_coords
            }
        ]

    """

    first_split_pt = np.where(np.abs(np.diff(y_var))>=np.pi)[0][0]

    split =[
        {
            xlabel : x_var[:first_split_pt],
            ylabel : y_var[:first_split_pt]
        }
    ] + [
                {
                    xlabel : x_var[(start+1):end],
                    ylabel : (y_var[(start+1):end])
                }
                for start, end in pairwise(np.where(np.abs(np.diff(y_var))>=np.pi)[0]) # segments whenever the rotation is of magnitude > np.pi
    ]
    return split