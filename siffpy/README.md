# SiffPy

The central SiffPy object: the `SiffReader`. I promise I'll document this soon. Each submodule has its own README as well, so for more info look there.
The submodules are made to be largely independent, and are separated to some extent by which libraries they rely on. `SiffTrac` and `SiffPlot`,
for example, rely on `HoloViews` and `Bokeh` for many functions, while the others do not.

## SiffTrac

Code for `FicTrac` files output by @tmohren's ROS implementations of `FicTrac`. Also contains functionality linking these
data to imaging data from the `.siff`.

## SiffPlot

Code for graphical analysis of `.siff` file image data. This is where ROI extraction, interactive visualization, heatmap plotting, etc.
takes place.

## SiffUtils

Code to assist the `SiffReader` object with interacting with `.siff` files (either shorthand to make code cleaner, or generic functions
that might be passed into the `siffreadermodule` object). Contains information about FLIM parameters, registration methods, and a few
simple math functions. 

## SiffMath

## SiffReaderModule

The `C/C++` code underlying direct interactions with the `.siff` filetype. It produces `numpy.ndarray` objects for image data 
as well as native Python objects for framewise (or experiment-wise) metadata. 