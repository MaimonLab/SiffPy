# SIFFMATH

Contains a bunch of numerical functions for analysis of data, mostly imaging.

Also contains custom `numpy` array extensions and `array`-like objects that use
the newer `numpy` dispatching mechanism.

# ARRAY SUBTYPES (TRACES)

The `siffmath` module contains multiple subclasses of the `numpy` `ndarray` object, 
contained within relevant submodules in a `traces.py` file.
This is primarily because much of these data should *mostly* behave like arrays, but
there are special considerations for each of them. For example, FLIM data should **not**
simply be added (or averaged). Instead it needs to be reweighted by each region-of-interest's
intensity values (or other parameters, depending on the FLIM metric being used) first, in a
series of element-wise operations. Implementing this with a pure Python class discards all of the
numerical and speed benefits of `numpy` and its toolkit, as well as making for more clumsy code.

The downside, of course, is that these become a little trickier to understand. Fortunately,
this is what documentation should be for! The types of arrays implemented here are documented
in this README, as well as the readmes of submodules (e.g. `flim` or `fluorescence`).

## FLUORESCENCE ARRAYS

The FluorescenceArrays are intended to deal with data like dF/F : transformations of
direct intensity measurements.

### **FluorescenceTrace**

The `FluorescenceTrace` class subclasses the `numpy.ndarray` class, with additional (mostly optional) arguments:

```
FluorescenceTrace(
    input_array : np.ndarray,
    method : str = None,
    normalized : bool = None,
    F : np.ndarray = None,
    F0 : np.ndarray = np.ndarray(None),
    max_val : np.ndarray = np.inf,
    min_val : np.ndarray = 0.0,
    angle : float = None,
    info_string : str = None
)
```

What each of these arguments are used for is detailed below.
What's relevant about these additional arguments is how they
transform with functions called on the `FluorescenceTrace`.

Some arguments are "VECTOR_PROPERTIES", which themselves are
transformed when `numpy` `ufuncs` are performed on the `FluorescenceTrace`.
These properties are: `F`, `F0`, `max_val`, `min_val`.

This way, if the `FluorescenceTrace` array is, for example,
scaled by some factor, the baseline `F` value is scaled
commensurately, as are the `max`, `min`, etc. 

Other properties are "LIST_PROPERTIES". These are *preserved*
if they're consistent across all `FluorescenceTrace` arrays
provided to the `ufunc`. Otherwise, they are lost. These
contain details about the generation of the `FluorescenceTrace`
that are meaningless if that detail is not shared by all of the
combined inputs.

**Arguments**

- `input_array : np.ndarray` :
  - A `numpy` array with which all outward functionality interacts directly.
  When you make a `FluorescenceTrace` out of an array, this argument is OBLIGATE.

- `method : str` : 
  - Used to define the approach used to generate the data stored here.
Some examples: `dF/F`, `raw_intensity`, `Photon counts`

- `normalized : bool` :
  - Used to define whether or not the data have been "normalized",
which usually is interpreted to mean "scaled from 0 to 1" or 
a similar approach. Useful to avoid accidentally mixing data that
has been processed in different ways.

- `F : np.ndarray` :
  - Stores the raw fluorescence data used to generate the
processed array in the main array context. Transforms alongside
the `FluorescenceTrace`.

- `F0 : np.ndarray` :
  - Stores a fluorescence value used to compute the measurement in
the `FluorescenceTrace` array. Usually is some baseline value (
e.g. mean value before some event, 5th percentile datum from the
raw `F` vector, etc.)

- `max_val : np.ndarray` :
  - Stores the `F` value that generated the maximal value in a
normalized `FluorescenceTrace`. So, for example, if the
`FluorescenceTrace` is normalized from 0 to 1, `max_val` would
correspond to the `dF/F` value at which the value 1 is achieved.

- `min_val : np.ndarray` :
  - As `max_val`, but for the 0 point in a normalization procedure.

- `angle : float` :
  - Generally a parameter reflecting some geometric property
of the trace within the image. A common example is the angular
coordinate of a particular wedge of the ellipsoid body, or a
particular column of the fan-shaped body.

- `info_string : str` :
  - A `str` used to contain extra information. No special
properties, just potentially useful for annotating data.

### **FluorescenceVector**

The `FluorescenceVector` is a special array constructed
only when a `FluorescenceTrace` is produced out of an iterable
whose elements are themselves all `FluorescenceTrace`s. Rather
than discard those elements' individual parameters (or
homogenize them), the `FluorescenceVector` preserves each of
their properties, either converting them into `list`s whose
elements correspond to each individual `FluorescenceTrace`'s
parameter, or into `numpy.ndarray` classes whose MAJOR axis
corresponds to the indexing of the iterable `FluorescenceTrace`
objects. All VECTOR_PROPERTIES are "numpify"ed, while all
LIST_PROPERTIES are "listify"ed.

When you index the major axis of a `FluorescenceVector`, it
returns a `FluorescenceTrace` rather than a `FluorescenceVector`
or a `np.ndarray`, as well as plucks out the corresponding major
axis element of the "numpify"ed attributes and that index from
the "listify"ed attributes.

## FLIM ARRAYS

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
`matplotlib`).

### **FlimTrace**

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
and `lifetime` attributes (the `lifetime` attribute is the primary array as well).

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
  - A `siffutils.FLIMParams` object for tracking the fit parameters used to estimate the `lifetime`

- `method : str`
  - A `str` tracking the method used to generate the lifetime estimate (e.g. `empirical lifetime`).

- `angle : float`
  - A coordinate used for tracking the geometrical location of the corresponding ROI in an image (see `FluorescenceTrace`).

- `info_string : str`
  - An unspecified `str` used to track additional information.

The `FlimTrace` also has a few useful properties:

- `FlimTrace.fluorescence` returns the `intensity` attribute as a `FluorescenceTrace`.

- `FlimTrace.lifetime` returns the primary array as a plain `np.ndarray`.

Functions on `FlimTrace` objects behave a little bit differently.

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
