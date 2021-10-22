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

