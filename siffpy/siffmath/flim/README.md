# SIFFMATH.FLIM

# FLIM ARRAYS

The main motivation for creating array subclasses: the annoyingness of
dealing with FLIM data consistently. Two arrays of lifetimes cannot
simply be added to one another, and averaging or grouping by ROI is made
much more of a hassle by this fact. Background pixels will have few photons
whose arrival times are likely uniformly distributed and those estimates alone
will greatly contaminate any pooling of lifetime information from the signal
pixels. The proper way to deal with these data is to pool all of the photon
arrival times together as if they came from one pixel and *then* compute
statistics on it. But this approach requires maintaining a histogram in memory
for each pixel (or else reading it from the file every time it's used), which
is hugely inefficient.

Fortunately, it's exactly equivalent (for empirical lifetime,
at least) to simply multiply each lifetime value by its corresponding intensity,
THEN add the arrays together, and finally divide by the sum of all pixels' intensities.
But if you just have an array of lifetime values and an array of pixel intensities,
it's difficult to keep track of this -- and even if you do, all of your numerical
processing libraries and tools WON'T know to do this! The `FlimTrace` exists to
resolve this headache: all the standard `numpy` functions should work on it, as
will any library that expects `array`-like data (e.g. `dask`, `HoloViews`, `napari`,
`matplotlib`). It's implemented using `numpy` itself, so it's fairly fast, though
not as fast as if it were implemented in the `C++` code directly. Still, compared to
file I/O this turns out to be a relatively small component of compute time. 

## **FlimTrace**

The `FlimTrace` is also an extended `np.ndarray`. Its signature looks like:
```
FlimTrace(
    input_array : np.ndarray,
    intensity : np.ndarray = None,
    confidence : np.ndarray = None,
    FLIMParams : FLIMParams = None,
    method : str = None,
    angle : float = None,
    info_string : str = None
)
```

While a `FlimTrace` *can* be initialized without `intensity`, it defeats some of the
purpose. Most `FlimTrace` operations require some interaction between the `intensity`
and `lifetime` attributes (the `lifetime` property returns a COPY of the primary array,
the `__array__()` method returns a reference to the array itself).

**Arguments**

- `input_array : np.ndarray`
  - An array, typically of `dtype` `float`, corresponding to some fluorescence lifetime
  metric, e.g. the empirical lifetime.

- `intensity : np.ndarray`
  - Must be same shape as `input_array`, this corresponds to the photon count number.
  If `None` is provided, this field will be filled with `np.nan`.

- `confidence : np.ndarray`
  - Not yet implemented. Will reflect confidence metrics in lifetime estimates

- `FLIMParams : FLIMParams`
  - A `siffpy.core.FLIMParams` object for tracking the fit parameters used to estimate the `lifetime`

- `method : str`
  - A `str` tracking the method used to generate the lifetime estimate (e.g. `empirical lifetime`).

- `angle : float`
  - A coordinate used for tracking the geometrical location of the corresponding ROI in an image (see `FluorescenceTrace`).

- `info_string : str`
  - An unspecified `str` used to track additional information.

The `FlimTrace` also has a few useful properties:

- `FlimTrace.fluorescence` returns the `intensity` attribute as a `FluorescenceTrace`.

- `FlimTrace.lifetime` returns a COPY of the primary array as a plain `np.ndarray`.

Functions on `FlimTrace` objects behave a little bit differently.

### CUSTOM FUNCTIONS

TODO: ANNOTATE

### CUSTOM UFUNCS

Most `ufunc`s follow a pattern with `FlimTrace`: if both arguments are a `FlimTrace`, they
return `NotImplemented`, otherwise they apply the `ufunc` to the **intensity** attribute
(NOT THE MAIN ARRAY!). This is because "multiplying" the lifetimes together doesn't make sense.
If you want to do that, get the raw array with the `lifetime` property!

- `np.add`
  - Addition on `FlimTrace`s is relatively simple. If a scalar or non-`FlimTrace` is added,
  only the `intensity` attribute will be modified (with the corresponding addition operation).
  If two `FlimTraces` are added, their `intensity`s are added, while their `__array__()` core
  attribute will be added pointwise as the following:
    - ``` returned_array_lifetime = (lifetime_1*intensity_1 + lifetime_2*intensity_2)/(intensity_1 + intensity_2)```

- `np.multiply`
  - Cannot multiply two `FlimTrace`s together. If multiplied by anything else, multiplication is performed
  on the `intensity` array alone.
    
TODO: ANNOTATE

### **FlimVector**

TODO: write this readme.

# ANALYSES

## Phase estimation

To get a list of available methods for estimating the phase of a vector signal, use
`phase_alignment_functions`, like so:

```
from siffpy import siffmath

siffmath.phase_alignment_functions() # will print in the console, notebook, etc.
```

All phase estimation methods follow these conventions (and may have additional arguments,
as can be discerned from their docstring with `help` or the printout of phase_alignment_functions):

- Require, as their first argument, a `np.ndarray` object whose shape is `(Dimension of ring, number of time bins)`.

- Accept, as a keyword argument, `estimate_error` to return error estimates along side the phase.

- Return, if `estimate_error` is `False` (always the default), a `np.ndarray` of shape `(number of time bins,)`.

- Return, if `estimate_error` is `True`, TO BE DETERMINED TODO: IMPLEMENT

## Fluorescence analysis

Most functions in the `fluorescence.py` submodule are accessible through `siffmath` directly, except for the
functions that are intended to be internal. To get a list of available methods, plus a print of their docstrings,
use `siffmath.string_names
