# SIFFPY

Python and C++ code for working with .tiffs (Tag Image File Format) generated by ScanImage and .siffs (SImple Flim Format) generated by my custom modified ScanImage (currently reachable at https://github.com/maimonlab/ScanImage2020). In my tests, handles files faster and more gracefully than ScanImage's tiffreader (possibly because it performs fewer checks -- my files are not as diverse).

### TODOS:
-    `pip` and `conda` installation style
-    Thorough docstrings for all classes and modules
-    Mask methods in `C` for fast multiple ROI reading, processing, etc.
-    Holoviews <--> Napari functionality (especially in `napari_fcns`) for all `Shapes` layer elements.
-    `DummySiffReader` to dupe `siffplot` machinery so that any previously-processed `numpy`
     or numpy-like array data can be fed to `siffplot` tools.
-    Siff2tiff
-    Add support for multi-ROI imaging.
-    Use cumulative bin occupancy not point estimates in chi-sq
-    Registration in pure C?
-    Example code (for every module and to go through a simple analysis pipeline)
-    Batching in registration alignment to take advantage of FFT scaling without massive memory issues
-    Improved regularization in registration, especially for systematically bad planes
-    Enable multithreaded and batched registration to minimize reads from disk but also minimize contiguous blocks of RAM.
-    Sphinx documentation + tutorial .ipynb
-    FlimArray object class implemented in C API to make FLIM data behave more like `numpy` arrays, instead
     of always keeping track of everything yourself (e.g. make them add like `numpy` arrays instead of you
     remembering to specify lines like `np.sum(intensity * lifetime)/np.sum(intensity)`).

## Installing SiffPy

Should work if you simply run `python setup.py install` in your conda environment of your choice. Requires numpy. Will update to be
more in tune with the more recent recommendations (i.e. everything purely through `pip` or `conda` install).

To really spell it out:

- Open a terminal and navigate to where you'd like to copy the SiffPy files with `cd` (e.g. `cd ~/Downloads`).
- Clone the repo a location of your choosing with `git clone https://github.com/maimonlab/SiffPy`
- Enter the newly created directory with `cd SiffPy`.
- Make sure you're in the environment you want, e.g. by typing `source activate flim`. You want to use one where the base Python install is Python3. I've been using `>3.9` with `futures` but none of that seems essential.
- Type `python setup.py install`.

This will also compile the C extension module `siffreadermodule` that does most of the heavy lifting, stick the library into your path for this environment, and then make the SiffPy Python code accessible.

Image registration is currently done in numpy but does not require loading the whole dataset into memory.
For now, it just does alignment using the intensity images, rather than
considering what source a photon is likely to have arrived from using FLIM.
So it can be run on laptops with relative ease.

### Dependencies 

- If you don't have numpy, or scipy it will complain during install and tell you to install `numpy` or `scipy` first (rather than downloading it yourself). To do so, you can either install it with `pip` by typing `pip install numpy` or with `conda` by typing `conda install -c anaconda numpy`. Uses only basic `numpy` includes, so version won't matter. `numpy` is necessary for `siffreadermodule` to compile, because many of its functions return pointers to `PyArrayObject`s. `scipy` is required for the `registration` submodule, which to me seems like a basic `SiffPy` functionality so I decided to make it a dependency.

- If you don't have `bokeh` or `holoviews` it will *warn* you during install, but will not prevent installation. These libraries are used by the `siffplot`
and by the `sifftrac` submodules, which (while written in `holoviews`) rely on a `bokeh` backend. This is optional so that if you want to use your own plotting
procedures, you're not forced to install a bunch of other libraries to your virtualenv.

- If you don't have `napari` it will also *warn* you during install, but will not prevent installation. At the time of writing, all `siffplot` functionality
*can* use `napari` and will *default* to `napari` if it's available, but will not complain if it cannot import `napari`. It will simply fall back to `holoviews`
and `bokeh` and function just fine. Still, `napari` is probably a better experience, and I may not always support both implementations forever, so it's
probably wiser to install `napari` (I just know some might not like it because it can be a bit of a headache when notebooks get involved).

- If you don't have `dask` it will also *warn* you, but this is only used for some `napari` functionality, so even if you're planning on using `napari`,
you may be able to get away without `dask` for some use cases. Still, this package is pretty useful for anything you might want to plot or analyse
dynamically, rather than importing the full array (many of these are hundreds of thousands of images, coming out to tens of GB of RAM). This is
another *strongly encouraged* type install.


## Using SiffPy

The primary object in `SiffPy` is the `SiffReader` class, which is usually imported with `from siffpy import SiffReader` at the start of a notebook. A `SiffReader` object handles I/O with a .siff or .tiff file to keep track of important file-specific variables as well as implements much of the boilerplate sort of code.

There are several submodules, each specializing in different tasks. Each has its own README.md file that explains more. Or... it will.
Most things are at least partly documented in the README files, and just about everything has fairly fleshed out docstrings.
I'll get around to ensuring everything is documented well soon though.

Each submodule is intended to function *in isolation and independent from* every other submodule, 
(though all rely on the `SiffReader`, and that in turn relies on the `siffutils` submodule),
so there are a few functions repeated in their respective
`utils` directories. This is 1) to avoid circular dependencies, 2) so that if one submodule is broken by
a user's update/modification, others need not fail.

### SiffPlot

This submodule handles visualization of data from .siff files. It relies on `Bokeh` and `HoloViews`, so you'll need to install those if you don't have them
if you plan to use this module. It has nicer and more flexible functionality if you use `napari`, so I recommend installing that (alongside `dask`). Because
`siffplot` is focused on image data, `napari` being a native image handler, rather than generic data visualization package, makes it function nicely here.

All of these are available on `conda` or `PyPI`.

### SiffTrac

This submodule handles alignment with FicTrac output datasets and visualization of the relationship
between FicTrac data and SiffReader data. This also relies on `Bokeh` and `HoloViews` for plotting functionality. But the
core objects (the `FicTracLog` and `LogInterpreter`) are independent of those packages.

### SiffUtils

This submodule handles functionality related to FLIM data, such as multiexponential fitting, and
functionality related to image processing, such as image registration.

### SiffMath

This submodule analyzes traces, sometimes produced by other modules (like the `SiffTrac` headings) and returns array-like objects.
Haven't decided, but it's plausible the plotter submodules might use this? I don't intend it to perform plotting itself, and so
will function even without `Holoviews` and `Bokeh` (at time of writing, Oct 21, 2021).

## Handling data

FLIM data is mostly naturally stored in sparse arrays: most pixels are not important, most histogram bins do not have many entries. But once you start building 512 by 512 pixel arrays, each of which have a FLIM measurement depth of 1024 bins, the array sizes get large quickly (one frame of this size would be 536 MB... acquiring at 30 Hz would give you a 16GB array for every one second of imaging). Most of SiffPy's functionality is performed lazily, avoiding loading arrays into memory unless some pixelwise relationship between several arrays is really needed. The `siffreader` C++ module mostly takes a `frames` or `frameslists` argument that allows pooling of frames by index, and the `siffpy` Python API does its best to hide all the nitty-gritty of that process from the user.

## More direct access to the data

The `siffreader` module contains lower-level access to the data, allowing you to directly get numpy arrays from .siff and .tiff files. To learn more, type `import siffreader; help(siffreader)` in your Python interpreter or in a Jupyter notebook.

Note:
So far I've only really been testing functionality in Jupyter notebooks. Note that if you use Jupyter lab, there are a few incompatibilities with the plotting libraries `matplotlib`, `bokeh`, and `holoviews`. However, the core code for extracting the data will be unaffected and relies ONLY on `numpy`.

## Understanding .siff files

.siff files are built to use the skeleton of a .tiff, but instead of each byte (or set of bytes) reflecting a pixel value,
they reflect a photon. As a result, all of the header and IFD structure of a .tiff is present (if you want to know what
those are, feel free to look them up), and so .tiff readers can help you navigate the files if you want to build your
own reader. The structure is as follows (TODO!!!).

### Reading individual frames.
Note that each individual IFD, and corresponding frame, has an additional tag, with the tag ID 907, called SiffCompress.
The SiffCompress flag is one byte, really one bit, reflecting whether that frame uses the compressed siff format or not.
It's a one byte tag because I expect that the future may hold other .siff formats, e.g. sparse count data (the standard
.siff tag but without arrival time data, so twice as compact, that lapses into .tiff storage when that becomes more
efficient), but probably not up to 255 of them.

__Uncompressed__

An uncompressed siff stores every photon in 8 bytes, with the 2 largest bytes giving the y coordinate, next 2
largest giving the x coordinate, and 4 smallest bytes giving the photon arrival time. So the 8 bytes
`00000000` `00000110` `00000000` `00111011` `00000000` `00000000` `00000000` `11111111` would refer to a photon arriving in the 255th time bin in the pixel
with y coordinate 6 and x coordinate 59. At present, only reading and writing with little endian is supported.
Yes, I know this is bad. It's a small tweak to the code to make it both-endian compatible and I will do it before
releasing a public-facing `1.0` version.

__Compressed__

A compressed siff stores every photon in 2 bytes, corresponding _only_ to the arrival time. This caps the number of
arrival bins permitted at 65535, which with the current finest resolution of the MultiHarp 150 (5 picoseconds) means 327 nanoseconds.
This is much longer than the time between 80 MHz laser pulses (12.5 nanoseconds) but puts a hard cap on the rep rate
of reduction by pulsepicking using this format, with that cap being 3.05 MHz. The pixel identity of each photon read
is stored at the start of the frame, with essentially what a normal tiff file would contain: a y-size-times-x-size element
array of `uint16`s, with each element corresponding to the number of photons in that pixel. The subsequent block of
bytes is each photon, starting with `y = 0` and increasing `x`, then resetting `x` to 0 and incrementing `y` to 1, and so
forth.

## Visualization and Plotting

There are multiple levels of plotting and analysis performed in the package, each on different types of data.
The `sifftrac` module handles logged data from FicTrac (see citation??) or the derived versions of FicTrac
implemented by @tmohren. The `siffplot` module handles image data from .siff and .tiff files, and produces
either images or analyzed plots, as well as providing an interface for selecting ROIs (or automatically segmenting).

For each of these, to get more information than provided here, navigate to the relevant directory (e.g. siffpy/sifftrac)
and read the `README.md` there.

### sifftrac, TracPlotters, and derived classes

The `sifftrac` module contains multiple classes, each derived from the `TracPlotter` and `FicTracLog`. The `TracPlotter`
file wraps `HoloViews` elements and has abstract methods designed for specialized plotting in derived classes, and is
initialized with a `FicTracLog` or list of `FicTracLog`s (or list of lists of `FicTracLog`s). The logic is to facilitate
composition of analyses on FicTrac data: derived classes of the `TracPlotter` can have specialized plotting functions,
but can also be composed to form joint figures (that can be examined in parallel or separately with their `Bokeh` toolbar(s)).
Composition also makes it easier to build `hv.Holomap` and `hv.DynamicMap` objects to explore the results of different
analyses in real time.

### siffplot, SiffPlotter, and SiffVisualizer

The `siffplot` module contains multiple classes designed to make it easy to see what's going in fluorescence images. The
`SiffVisualizer` class produces `hv.DynamicMap` objects or `napari.Viewer`s that read from disk to produce images or stacks of images with
accompanying quick analyses performed on each (e.g. pooling, masking). The `SiffPlotter` takes a .siff file and produces
more "analyzed" output, e.g. segmentation into regions of interest or heatmaps of activity across ROIs. Both are coupled
to a `SiffReader` object which retains access to (and performs actual reading of) the .siff file.

Warning: `SiffPlotter`s are `holoviews` only, but `SiffVisualizer`s use either `napari` or `holoviews` (ideally will always
implement both, but no promises!). Some builds of Jupyter, IPython, `holoviews` and `napari` do not play nice together and
you'll experience strange lags in cell evaluation because `holoviews` and `napari` both interface with Jupyter's event loop.