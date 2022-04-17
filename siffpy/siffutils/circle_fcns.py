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

def circ_interpolate_between_endpoints(x_samp : np.ndarray, endpts_x : np.ndarray, endpts_y : np.ndarray):
    """
    Interpolates between arrays of endpoints in a manner consistent with a circular variable,
    then samples at the requested intermediate values 'x_samp'. For every x_samp, the two other
    arguments must contain two bounding points
    
    Arguments
    ---------

    x_samp : np.ndarray

        The x-coordinates on which the interpolation will be evaluated. Must be of size
        (N,) or (N,1)

    endpts_x : np.ndarray

        An array with one of its dimensions of length 2, corresponding to points between
        which to interpolate.

    """
    # a few formatting checks
    if not (
        any(k == 2 for k in endpts_x.shape) or
        any(k == 2 for k in endpts_y.shape)
    ):
        raise ValueError("Must pass arrays with at least one dimension of length 2")
    
    if not (endpts_x.shape[0] == 2):
        endpts_x = endpts_x.T
    
    if not (endpts_y.shape[0] == 2):
        endpts_y = endpts_y.T
    
    if not (endpts_x.shape[1] == x_samp.shape[0]) and (endpts_y.shape[1] == x_samp.shape[0]):
        raise ValueError("Must pass two endpoints to interpolate between for every sampled point")
    # how much closer to the left endpoint than the right
    frac_in = (x_samp-endpts_x[0])/(endpts_x[1]-endpts_x[0])
    
    # figure out the circular distance between the y points
    naive_diff = np.angle(np.exp(endpts_y[1,:]*1j)/np.exp(endpts_y[0,:]*1j)) 

    return (
        np.angle(
            np.exp(endpts_y[0,:]*1j)*
            np.exp(frac_in*naive_diff*1j)
        )
    ).flatten() # offset by the difference, then recircularize

def split_angles_to_list(x_var : np.ndarray, y_var : np.ndarray)->list[tuple]:
    """
    Returns a list of tuples that can be passed to a HoloViews Path object.

    Splits circular variables up whenever there's a shift > pi in magnitude.

    ARGUMENTS
    ---------
    
    x_var : 
        
        The independent variable (usually the kdim) to pass to Holoviews

    y_var :

        The dependent variable (usually to be the vdim) to pass to Holoviews

    RETURNS
    -------

    split : list[tuple]

        A list of tuples corresponding to individual line segments before they need to be
        made disjoint. Schematized as:

        [
            (
                consecutive_x_coords,
                consecutive_y_coords
            )
        ]
    """

    first_split_pt = np.where(np.abs(np.diff(y_var))>=np.pi)[0][0]

    split =[
        (
            x_var[:first_split_pt],
            y_var[:first_split_pt]
        )
    ] + [
        (
            x_var[(start+1):min(end+1,len(x_var))],
            y_var[(start+1):min(end+1,len(y_var))]
        )
        for start, end in pairwise(np.where(np.abs(np.diff(y_var))>=np.pi)[0])  
    ]
    return split

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
                    xlabel : x_var[(start+1):min(end+1,len(x_var))],
                    ylabel : (y_var[(start+1):min(end+1,len(y_var))])
                }
                for start, end in pairwise(np.where(np.abs(np.diff(y_var))>=np.pi)[0]) # segments whenever the rotation is of magnitude > np.pi
    ]
    return split