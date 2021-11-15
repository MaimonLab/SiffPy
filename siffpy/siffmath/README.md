# SIFFMATH

Contains a bunch of numerical functions for analysis of data, mostly imaging.

TODO: write this readme.

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
functions that are intended to be internal. It's organized like this because I want it to work even if you don't
have `Holoviews` installed, and so some functions need alternative implementations. So `fluorescence.py` checks
if it can `import Holoviews`, and then defines functions differently based on whether it can or can't do so.
Then `siffmath`'s `__init__.py` imports all functions from `fluorescence.py`, but `fluorescence.py`'s `__all__`
conceals the interal functions that are differentially implemented and provides a common interface for both.

