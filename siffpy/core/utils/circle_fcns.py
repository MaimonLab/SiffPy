# Functions for circularizing floats and ints
from typing import Any, Tuple

import numpy as np

def circ_d(x : float, y : float, rollover : float)->float:
    """Wrapped-around distance between x and y"""
    return ((x-y + rollover/2) % rollover) - rollover/2

def re_circ(x : float, rollover : float) -> float:
    """ Takes de-circularized data and reverts it to circularized """
    return (x + rollover) % rollover

def roll_d(roll1 : Tuple[float, float], roll2: Tuple[float,float], rollover_y: float, rollover_x : float)->float:
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

def circ_corr(
        x : np.ndarray,
        y : np.ndarray,
        axis : int = 0,
        method : str = 'Fisher'
    )->float:
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

    Arguments
    ---------

    x : np.ndarray

        One of the two arrays of circular variables to correlate (in radians
        doesn't matter if it starts at 0 or -pi)

    y : np.ndarray

        The other of the two arrays of circular variables to correlate (in radians
        doesn't matter if it starts at 0 or -pi)

    axis : int = 0

        The axis along which to take the correlation (i.e. the direction being summed).
        Defaults to 0.

    method : str = "Fisher"

        The method to use to compute the correlation. Options are "Fisher" and
        "Jammalamadaka" (also accepts "Pearson-sine" as an alias for "Jammalamadaka").
        Defaults to "Fisher".

    Returns
    -------

    rho : float

        The circular correlation between x and y

    Examples
    --------
    ```python
    import numpy as np
    from siffpy.core.utils.circle_fcns import circ_corr

    x = np.linspace(0,2*np.pi,1000)
    y = np.linspace(1,1+2*np.pi,1000)

    circ_corr(x,y)
    >>> 1.0000000000000004
    ```

    Let's try random noise

    ```python
    import numpy as np
    from siffpy.core.utils.circle_fcns import circ_corr

    np.random.seed(0)
    x = np.random.rand(1000)*2*np.pi
    y = np.random.rand(1000)*2*np.pi

    circ_corr(x,y)
    >>> -0.0005595881845383295
    ```

    And noise on top of a signal


    ```python
    import numpy as np
    from siffpy.core.utils.circle_fcns import circ_corr

    np.random.seed(0)

    # Random steps
    dx = np.random.rand(1000)*0.01

    # accumulate
    x = np.cumsum(dx)

    # Add noise
    y = x + np.random.rand(1000)

    circ_corr(x,y)
    >>> 0.9131513426682732
    ```
    """
    expd_1 = np.exp(1j*x)
    expd_2 = np.exp(1j*y)

    return circ_corr_complex(expd_1, expd_2, axis=axis, method = method)

def circ_corr_complex(
        x : np.ndarray,
        y : np.ndarray,
        axis : int = 0,
        method : str = "Fisher"
    )->float:
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

    method : str = "Fisher"

        The method to use to compute the correlation. Options are "Fisher" and
        "Jammalamadaka" (also accepts "Pearson-sine" as an alias for "Jammalamadaka").
        Defaults to "Fisher".
    """
    if method == "Fisher":
        return circ_corr_complex_fisher(x,y,axis)
    elif method in ["Jammalamadaka", "Pearson-sine"]:
        return circ_corr_complex_jl(x,y,axis)
    
    raise ValueError(
        "Invalid method passed to circ_corr_complex. Must be"
        "either 'Fisher' or 'Jammalamadaka'"
        )
    
def circ_corr_complex_jl(
        x : 'np.ndarray[Any, np.dtype[np.complex128]]',
        y : 'np.ndarray[Any, np.dtype[np.complex128]]',
        axis : int = 0,
    )->float:
    """
    Presumes x and y are already complex numbers on the unit circle!
    This one is even faster because it skips the exp step. Recommended
    if you're going to run the function many times that you put it in
    the complex plane first and then execute this.

    Compute circular correlation a la Jammalamadaka et al (2001)

    rho = Sum( sin(x - E[x]) * sin(y - E[y])) /
    sqrt( Sum(sin(x - E[x])**2) * Sum(sin(y - E[y])**2) )

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
    x_zeroed, y_zeroed = x/np.mean(x, axis=axis), y/np.mean(y, axis=axis)
    return np.sum(x_zeroed.imag * y_zeroed.imag, axis=axis) / np.sqrt(
        np.sum(x_zeroed.imag**2, axis=axis) * np.sum(y_zeroed.imag**2, axis=axis)
    )

def circ_corr_complex_fisher(
        x : 'np.ndarray[Any, np.dtype[np.complex128]]',
        y : 'np.ndarray[Any, np.dtype[np.complex128]]',
        axis : int = 0,
    )->float:
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

def running_circ_corr(
        x : np.ndarray,
        y : np.ndarray,
        window_width : int,
        axis : int = 0,
        method : str = "Fisher"
        )->np.ndarray:
    """
    Takes two arrays of circular numbers and computes the circular correlation
    between them in a sliding window fashion. The returned array is window_width elements
    shorter than the input array. The correlation is _centered_, meaning it will start on
    the window_width//2-th element and end on the len - window_width//2-th element.

    Arguments
    ---------
    x : np.ndarray
        
        Values of -pi to +pi (or 0 to 2 pi, just has to be radians)

    y: np.ndarray

        Values of -pi to +pi (or 0 to 2 pi, just has to be radians)

    window_width : int

        The width of the sliding window in numbers of entries

    axis : int = 0

        The axis along which to take the correlation (i.e. the direction being summed).
        Defaults to 0.

    method : str = "Fisher"

        The method to use to compute the correlation. Options are "Fisher" and
        "Jammalamadaka" (also accepts "Pearson-sine" as an alias for "Jammalamadaka").
        Defaults to "Fisher".

    Returns
    -------
    circ_corrs : np.ndarray

        Note that the shape will be `len(x) - window_width` along the axis
        that the window is being taken.

    Examples
    --------

    ```python
    import numpy as np
    from siffpy.core.utils.circle_fcns import running_circ_corr

    corr = running_circ_corr(x,y,10)

    print(
        corr.shape,
        np.all(np.abs(corr - 1) < 0.01)
    )

    >>> ((990,), True)
    ```

    Let's try random noise

    ```python
    import numpy as np
    from siffpy.core.utils.circle_fcns import running_circ_corr

    np.random.seed(0)
    x = np.random.rand(1000)*2*np.pi
    y = np.random.rand(1000)*2*np.pi

    corr = running_circ_corr(x,y,100)

    print(
        corr.shape,
        np.all(np.abs(corr - 0) < 0.1)
    )
    >>> ((900,), True)
    ```

    Notably, if the window is very short, even random noise will 
    show a pretty strong correlation!

    ```python
    import numpy as np
    from siffpy.core.utils.circle_fcns import running_circ_corr

    np.random.seed(0)
    x = np.random.rand(50)*2*np.pi
    y = np.random.rand(50)*2*np.pi

    corr = running_circ_corr(x,y,10)

    print(
        corr.shape,
        np.all(np.abs(corr - 0) < 0.1)
    )

    >>> ((40,), False)
    ```
    """
    x = np.exp(1j*x)
    y = np.exp(1j*y)
    return running_circ_corr_complex(x,y,window_width,axis,method)

def running_circ_corr_complex(
        x : 'np.ndarray[Any, np.dtype[np.complex128]]',
        y : 'np.ndarray[Any, np.dtype[np.complex128]]',
        window_width : int,
        axis : int = 0,
        method : str = "Fisher"
        )->np.ndarray:
    """
    Takes two arrays of complex numbers and computes the circular correlation
    between them in a sliding window fashion. The returned array is window_width elements
    shorter than the input array. The correlation is _centered_, meaning it will start on
    the window_width//2-th element and end on the len - window_width//2-th element.

    Arguments
    ---------
    x : np.ndarray

    y: np.ndarray

    window_width : int

        The width of the sliding window in numbers of entries
    
    axis : int = 0

        The axis along which to take the correlation (i.e. the direction being summed).
        Defaults to 0.

    method : str = "Fisher"

        The method to use to compute the correlation. Options are "Fisher" and
        "Jammalamadaka" (also accepts "Pearson-sine" as an alias for "Jammalamadaka").
        Defaults to "Fisher".
    
    Returns
    -------
    circ_corrs : np.ndarray

    Examples
    --------

    ```python
    import numpy as np
    from siffpy.core.utils.circle_fcns import running_circ_corr_complex

    x = np.linspace(0,2*np.pi,1000)
    y = np.linspace(1,1+2*np.pi,1000)

    corr = running_circ_corr_complex(np.exp(1j*x),np.exp(1j*y),10)

    print(
        corr.shape,
        np.all(np.abs(corr - 1) < 0.01)
    )

    >>> ((990,), True)
    """ 

    if method in ("Fisher", 'fisher'):
        return running_circ_corr_complex_fisher(x,y,window_width,axis)
    elif method in ("jammalamadaka", "Jammalamadaka", "Pearson-sine"):
        return running_circ_corr_complex_jl(x,y,window_width,axis)
    
    raise ValueError(
        "Invalid method passed to running_circ_corr_complex. Must be"
        "either 'Fisher' or 'Jammalamadaka'"
        )

def running_circ_corr_complex_jl(
        x : 'np.ndarray[Any, np.dtype[np.complex128]]',
        y : 'np.ndarray[Any, np.dtype[np.complex128]]',
        window_width : int,
        axis : int = 0,
        )->'np.ndarray[Any, np.dtype[np.float64]]':
    """
    Takes two arrays of complex numbers and computes the circular correlation
    between them in a sliding window fashion. The returned array is window_width elements
    shorter than the input array. The correlation is _centered_, meaning it will start on
    the window_width//2-th element and end on the len - window_width//2-th element.

    Arguments
    ---------
    x : np.ndarray

    y: np.ndarray

    window_width : int

        The width of the sliding window in numbers of entries
    
    axis : int = 0

        The axis along which to take the correlation (i.e. the direction being summed).
        Defaults to 0.
    
    Returns
    -------
    circ_corrs : np.ndarray
    """
    x_zeroed, y_zeroed = x/np.mean(x, axis=axis), y/np.mean(y, axis=axis)
    
    numerator = np.cumsum(x_zeroed.imag * y_zeroed.imag, axis=axis)
    x_sq, y_sq = np.cumsum(x_zeroed.imag**2, axis=axis), np.cumsum(y_zeroed.imag**2, axis=axis)

    return (
        (numerator[window_width:] - numerator[:-window_width])
        / np.sqrt(
            (x_sq[window_width:] - x_sq[:-window_width]) *
            (y_sq[window_width:] - y_sq[:-window_width])
        )
    )

def running_circ_corr_complex_fisher(
        x : 'np.ndarray[Any, np.dtype[np.complex128]]',
        y : 'np.ndarray[Any, np.dtype[np.complex128]]',
        window_width : int,
        axis : int = 0,
        )->np.ndarray:
    """
    Takes two arrays of complex numbers and computes the circular correlation
    between them in a sliding window fashion. The returned array is window_width elements
    shorter than the input array. The correlation is _centered_, meaning it will start on
    the window_width//2-th element and end on the len - window_width//2-th element.

    Arguments
    ---------
    x : np.ndarray

    y: np.ndarray

    window_width : int

        The width of the sliding window in numbers of entries
    
    axis : int = 0

        The axis along which to take the correlation (i.e. the direction being summed).
        Defaults to 0.
    
    Returns
    -------
    circ_corrs : np.ndarray
    """

    plus = x*y
    minus = x/y

    plus_cumsum = np.cumsum(plus,axis = axis)
    minus_cumsum = np.cumsum(minus, axis = axis)
    run_plus = plus_cumsum[window_width:] - plus_cumsum[:-window_width]
    run_minus = minus_cumsum[window_width:] - minus_cumsum[:-window_width]

    run_num = (
        (run_minus * np.conjugate(run_minus)) -
        (run_plus * np.conjugate(run_plus))
    )

    x_cumsum = np.cumsum(x**2, axis=0)
    y_cumsum = np.cumsum(y**2, axis=0)

    run_xcs = x_cumsum[window_width:] - x_cumsum[:-window_width]
    run_ycs = y_cumsum[window_width:] - y_cumsum[:-window_width]

    run_den = np.sqrt(
        (window_width**2 - run_xcs*np.conjugate(run_xcs))*
        (window_width**2 - run_ycs*np.conjugate(run_ycs))
    )

    return (run_num/run_den).real
    

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