# SiffPy

The central SiffPy object: the `SiffReader`. I promise I'll document this soon. The only thing I expect all `SiffPy` users
to need is `siffpy.SiffReader`, its obligate helper module `siffreadermodule`, and the `siffpy.core` tools. The `siffmath` package contains
some more elaborate tools for dealing with fluorescence and FLIM data.

## Core

The `core` submodule contains functions for reading `.siff` files and
converting them into `numpy` arrays. Its functionality revolves around
the `SiffReader` class, which wraps the various methods of the
`siffreadermodule` and returns nicely formatted arrays.

## SiffMath

Numerical code that does not depend on or directly interact with any plotting functionality. Operates almost exclusively with
`numpy` arrays and creates several `np.ndarray` subclasses that track accompanying metadata of vector-type data usefully (for 
example, adding and subtracting scalars or summing two such arrays).