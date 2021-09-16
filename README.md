# SIFFPY

Python and C++ code for working with .tiffs (Tag Image File Format) generated by ScanImage and .siffs (SImple Flim Format) generated by my custom modified ScanImage (currently reachable at https://github.com/maimonlab/ScanImage2020). In my tests, handles files faster and more gracefully than ScanImage's tiffreader (possibly because it performs fewer checks -- my files are not as diverse).

### TODOS:
-    Add support for multi-ROI imaging.
-    Use cumulative bin occupancy not point estimates in chi-sq
-    Registration in pure C?
-    Update example code to include image registration + diagnostics
-    Batching in registration alignment to take advantage of FFT scaling without massive memory issues
-    Improved regularization in registration, especially for systematically bad planes
-    Speed up registration, e.g. move the convolution into the Fourier domain (might speed things up a lot)
-    Sphinx documentation + tutorial .ipynb

## Installing SiffPy

Should work if you simply run `python setup.py install` in your conda environment of your choice. Requires numpy. To really spell it out:

- Open a terminal and navigate to where you'd like to copy the SiffPy files with `cd` (e.g. `cd ~/Downloads`).
- Clone the repo a location of your choosing with `git clone https://github.com/maimonlab/SiffPy`
- Enter the newly created directory with `cd SiffPy`.
- Make sure you're in the environment you want, e.g. by typing `source activate flim`. You want to use one where the base Python install is Python3. I've been using `>3.9` with `futures` but none of that seems essential.
- Type `python setup.py install`.

If you don't have numpy, it will complain during install and tell you to install `numpy` first (rather than downloading it yourself). To do so, you can either install it with `pip` by typing `pip install numpy` or with `conda` by typing `conda install -c anaconda numpy`. Uses only basic `numpy` includes, so version won't matter. 

This will compile the C extension module `siffreadermodule` that does most of the heavy lifting, stick the library into your path for this environment, and then make the SiffPy Python code accessible.

Image registration is currently done in numpy but does not require loading the whole dataset into memory.
For now, it just does alignment using the intensity images, rather than
considering what source a photon is likely to have arrived from using FLIM.

## Using SiffPy

The primary object in `SiffPy` is the `SiffReader` class, which is usually imported with `from siffpy import SiffReader` at the start of a notebook. A `SiffReader` object handles I/O with a .siff or .tiff file to keep track of important file-specific variables as well as implements much of the boilerplate sort of code.

There are several submodules, each specializing in different tasks. Each has its own README.md file that explains more. Or... it will. I'll get around to it.

### SiffPlot

This submodule handles visualization of data from .siff files.

### SiffTrac

This submodule handles alignment with FicTrac output datasets and visualization of the relationship
between FicTrac data and SiffReader data.

### SiffUtils

This submodule handles functionality related to FLIM data, such as multiexponential fitting, and
functionality related to image processing, such as image registration.

### Handling data

FLIM data is mostly naturally stored in sparse arrays: most pixels are not important, most histogram bins do not have many entries. But once you start building 512 by 512 pixel arrays, each of which have a FLIM measurement depth of 1024 bins, the array sizes get large quickly (one frame of this size would be 536 MB... acquiring at 30 Hz would give you a 16GB array for every one second of imaging). Most of SiffPy's functionality is performed lazily, avoiding loading arrays into memory unless some pixelwise relationship between several arrays is really needed. The `siffreader` C++ module mostly takes a `frames` or `frameslists` argument that allows pooling of frames by index, and the `siffpy` Python API does its best to hide all the nitty-gritty of that process from the user.

### More direct access to the data

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

__Uncompressed__

An uncompressed siff stores every photon in 8 bytes, with the 2 largest bytes giving the y coordinate, next 2
largest giving the x coordinate, and 4 smallest bytes giving the photon arrival time. So the 8 bytes
`00000000` `00000110` `00000000` `00111011` `00000000` `00000000` `00000000` `11111111` would refer to a photon arriving in the 255th time bin in the pixel
with y coordinate 6 and x coordinate 59.

__Compressed__

A compressed siff stores every photon in 2 bytes, corresponding _only_ to the arrival time. This caps the number of
arrival bins permitted at 65535, which with the current finest resolution of the TimeHarp 150 means 327 nanoseconds.
This is much longer than the time between 80 MHz laser pulses (12.5 nanoseconds) but puts a hard cap on the rep rate
of reduction by pulsepicking using this format, with that cap being 3.05 MHz. The pixel identity of each photon read
is stored at the start of the frame, with essentially what a normal tiff file would contain: a y-size-times-x-size element
array of `uint16`s, with each element corresponding to the number of photons in that pixel. The subsequent block of
bytes is each photon, starting with `y = 0` and increasing `x`, then resetting `x` to 0 and incrementing `y` to 1, and so
forth. 