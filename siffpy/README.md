# SiffPy

The central SiffPy object: the `SiffReader`. I promise I'll document this soon. Each submodule has its own README as well, so for more info look there.
The submodules are made to be largely independent, and are separated to some extent by which libraries they rely on. `SiffTrac` relies on `pandas`, and
`SiffPlot` relies on `HoloViews` and `Bokeh` for many functions, while the others do not. The only thing I expect all `SiffPy` users
to need is `siffpy.SiffReader`, its obligate helper module `siffreadermodule`, and the `siffpy.core` tools. The `siffmath` package contains
some more elaborate tools for dealing with fluorescence and FLIM data.

## Core

The `core` submodule contains functions for reading `.siff` files and
converting them into `numpy` arrays. Its functionality revolves around
the `SiffReader` class, which wraps the various methods of the
`siffreadermodule` and returns nicely formatted arrays.

## SiffTrac

Code for `FicTrac` files output by @tmohren's ROS implementations of `FicTrac`. Also contains functionality linking these
data to imaging data from the `.siff`. Used by the `SiffPlot` submodule but only in its `TracPlot` classes. Currently uses `pandas`, soon to migrate to `polars`

## SiffPlot

Code for graphical analysis of `.siff` file image data. This is where ROI extraction, interactive visualization, heatmap plotting, etc.
takes place.

## SiffMath

Numerical code that does not depend on or directly interact with any plotting functionality. Operates almost exclusively with
`numpy` arrays and creates several `np.ndarray` subclasses that track accompanying metadata of vector-type data usefully (for 
example, adding and subtracting scalars or summing two such arrays).