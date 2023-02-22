# Functions for circularizing floats and ints
from itertools import tee
from tkinter import Y

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

def circ_diff(arr : np.ndarray)->np.ndarray:
    """
    Takes the difference of presumed circular variables.
    Returned array is 1 element shorter than the input array.

    Arguments
    --------

    arr : np.ndarray

        A 1d array circular variable

    Returns
    -------

    diff : np.ndarray

        The circular difference between successive time points.
        Length of the array is 1 
    """

    return np.angle(np.exp(arr[1:]*1j)/np.exp(arr[:-1]*1j))

def circ_unwrap(arr: np.ndarray, offset : float = 0.0)->np.ndarray:
    """
    Takes a circular variable and converts it to an unwrapped circular variable
    (i.e. sums the cumulative differences, so that instead of ranging over an 
    interval of width 2*pi, it ranges from -inf to +inf). Sets the first point
    to offset

    Arguments
    --------

    arr : np.ndarray

        A circular variable with periodicity 2*pi.

    offset : float = 0.0

        What the value of the first point should be
    """
    return np.insert(np.cumsum(circ_diff(arr)),0,offset)

def circ_corr(x : np.ndarray, y : np.ndarray, axis : int = 0)->float:
    """
    Warning: recommend putting your angles in the complex plane
    FIRST and using circ_corr_complex, because putting your time series
    into the complex plane is the most expensive operation here.
    But this function is for if you don't want to deal with complex numbers
    in your code. Still runs faster than if you do it purely with real numbers.

    Compute circular correlation a la Green, Adachi, Maimon 2017,
    itself borrowing a circular correlation toolkit from 
    Jessica B. Hamrick and Peter W. Battaglia, who in turn
    built their function after the MATLAB circ functions,
    themselves deriving this function from Fisher et al. 1983.

    rho = 2*( (E[cos(x-y)]**2 + E[sin(x-y)]**2) - (E[cos(x+y)]**2 + E[sin(x+y)]**2) ) / Z

    Z is the normalization constant

    This measure reflects the difference between an estimated positive exact relationship
    between x and y vs. an estimated negative relationship. Each term is the expected magnitude of
    a resultant vector made from either x - y (a constant if there's a positive relationship
    between the two) or x + y (a constant if there's a negative relationship between the two.)
    The expected magnitude of the constant vector is 1, and so if rho is approximately 0 if the
    two have no relationship, 1 if the two have a constant positive relationship, and -1 if
    the two have a constant negative offset

    TODO: figure out good bounds
    on the error.

    x and y are presumed to be in radians, but there's no
    need to use mod.

    Implemented with complex numbers instead of cos and sine as in the original
    paper because it runs faster and is far more elegant.

    ARGUMENTS
    ---------

    x : np.ndarray

        One of the two arrays of circular variables to correlate

    y : np.ndarray

        The other of the two arrays of circular variables to correlate

    axis : int = 0

        The axis along which to take the correlation (i.e. the direction being summed).
        Defaults to 0.
    """
    expd_1 = np.exp(1j*x)
    expd_2 = np.exp(1j*y)

    return circ_corr_complex(expd_1, expd_2, axis=axis)

def circ_corr_complex(x : np.ndarray, y : np.ndarray, axis : int = 0)->float:
    """
    Presumes x and y are already complex numbers on the unit circle!
    This one is even faster because it skips the exp step. Recommended
    if you're going to run the function many times that you put it in
    the complex plane first and then execute this.

    Compute circular correlation a la Green, Adachi, Maimon 2017,
    itself borrowing a circular correlation toolkit from 
    Jessica B. Hamrick and Peter W. Battaglia, who in turn
    built their function after the MATLAB circ functions,
    themselves deriving this function from Fisher et al. 1983.

    rho = 2*( (E[cos(x-y)]**2 + E[sin(x-y)]**2) - (E[cos(x+y)]**2 + E[sin(x+y)]**2) ) / Z

    Z is the normalization constant

    This measure reflects the difference between an estimated positive exact relationship
    between x and y vs. an estimated negative relationship. Each term is the expected magnitude of
    a resultant vector made from either x - y (a constant if there's a positive relationship
    between the two) or x + y (a constant if there's a negative relationship between the two.)
    The expected magnitude of the constant vector is 1, and so if rho is approximately 0 if the
    two have no relationship, 1 if the two have a constant positive relationship, and -1 if
    the two have a constant negative offset

    TODO: figure out good bounds
    on the error.

    Implemented with complex numbers instead of cos and sine as in the original
    paper because it runs faster and is far more elegant.

    ARGUMENTS
    ---------

    x : np.ndarray

        One of the two arrays of circular variables to correlate (in form exp(1j*theta_1))

    y : np.ndarray

        The other of the two arrays of circular variables to correlate (in form exp(1j*theta_2))

    axis : int = 0

        The axis along which to take the correlation (i.e. the direction being summed).
        Defaults to 0.
    """
    
    plus = x*y
    minus = x/y
    
    # The first half of the term is the prediction of a POSITIVE association between x and y, i.e. x ~ y + alpha,
    # the second half is the prediction of a NEGATIVE association between x and y, i.e. x ~ -y + alpha.
    # Each term is the magnitude of the resultant mean vector of the sum vs. the difference of the two series

    plussum = np.sum(plus,axis = axis)
    minussum = np.sum(minus, axis = axis)
    
    numerator = (
        minussum* np.conjugate(minussum) -
        plussum * np.conjugate(plussum)
    )

    xsum = np.sum(x**2, axis=axis)
    ysum = np.sum(y**2, axis=axis)

    # normalization factor
    
    denominator = np.sqrt(
        (x.shape[axis]**2 - xsum*np.conjugate(xsum)) *
        (y.shape[axis]**2 - ysum*np.conjugate(ysum))
    )
    
    return np.real(numerator/denominator)

def circ_corr_non_parametric(x : np.ndarray, y : np.ndarray)->float:
    """
    Compute a ranked circular correlation from Fisher et al. 1983.

    P = (4/n^2)( Sum_{1<=i<j<=n} sin(2*pi*(r_i-r_j)/n)*sin(2*pi(s_i-s_j)/n) )
    with {r_k} the ranks of the elements of x and {s_k} the ranks of the elements of y

    This measure bears the same relationship with the circ_corr as Pearson's r has with
    Spearmann's rho. With _uniform_ (not von Mises) marginals, this asymptotically converges
    to the distribution of circ_corr.

    TODO: figure out good bounds
    on the error.

    Can't help but suspect there's an even more elegant way to
    implement this using complex numbers. There's never a time that
    circular functions are better implemented in R than in C.
    """

    raise NotImplementedError("Too lazy.")


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

def circ_interpolate_to_sample_points(x_samp : np.ndarray, x_axis : np.ndarray, y_axis: np.ndarray)->np.ndarray:
    """
    Takes two arrays corresponding to the x_axis and y_axis of some data, and circ-linearly interpolates
    the y_axis to sample at the values of x_samp.

    Return range is -pi to +pi

    Returns
    -------

    interpolated_y : np.ndarray

        Same shape as x_samp
    """

    # get the nearest x_axis points to interpolate between
    x_axis_endpts_idxs = [np.argpartition(np.abs(x_samp[x]-x_axis),2)[:2] for x in range(len(x_samp))]
    x_axis_endpts = x_axis[x_axis_endpts_idxs]
    y_axis_endpts = y_axis[x_axis_endpts_idxs]

    # now interpolate between the endpoints
    return circ_interpolate_between_endpoints(
        x_samp,
        x_axis_endpts,
        y_axis_endpts
    )

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