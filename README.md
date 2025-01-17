# SIFFPY

For the latest documentation and guides, check the `readthedocs`:

https://siffpy.readthedocs.io/en/latest/

`Python`, `Rust`, and `C++` code for working with .tiffs (Tag Image File Format) generated by ScanImage
and .siffs (SImple Flim Format / ScanImage-FLIM format) generated by my 
custom modified ScanImage (currently reachable as a bunch of distributed repos but the core
code being in `PicoQuantScanImageTools`).

With the current `corrosiff` backend, this and all future versions (until PyO3 changes)
will not be compatible with `PyPy` interpreters. This is only because `PyPy` is not
supported by the `rust-numpy` tools, and manually bridging the `Rust` library to `Python`
would be a nightmare.

### TODOS:
-   In-place operations should be performed correctly in `FlimTrace`, e.g.
`my_trace -= 2` or `my_trace -= another_trace`. Currently `NotImplemented` which
is very very bad behavior!!
-   Document the `corrosiffpy` wrapper on the `docs`.
-   IEM for lifetime estimates as an option (seems like it will always be slow though??)
-   1d-numpy array returning methods in `C++` for fast pixelwise within-ROI analyses
-   Convert FLIM fitting to `mystic` or another more robust solver than `trust-constr`?
-   Add more explicit support for multi-ROI imaging. Currently it stacks the
ROIs all into one array and it's the user's responsibility to discern which pixels
are which ROI. 
-   Use cumulative bin occupancy not point estimates in chi-sq
-   Example code (for every module and to go through a simple analysis pipeline)
-   Improved regularization in registration, especially for systematically bad planes

## Installing SiffPy

In some cases, if you're on the Rockefeller University network, and if you prefer to
manage with `conda`, and if the latest version has been compiled and packaged, you
can install with `conda` using a custom channel. You can install it with

```
conda install siffpy -c <path_to_maimondata01>/maimondata01/lab_resources/maimon-forge
```

Otherwise, it should install if you simply run `python3 -m pip install .` in your conda
environment of your choice. Requires `numpy`.

To really spell it out:

- Open a terminal and navigate to where you'd like to copy the SiffPy files with `cd` (e.g. `cd ~/Downloads`).
- Clone the repo a location of your choosing with `git clone https://github.com/maimonlab/SiffPy`
- Enter the newly created directory with `cd SiffPy`.
- Make sure you're in the environment you want, e.g. by typing `source activate flim`. You want to use one where the base Python install is Python3. I've been using `>3.9` with `futures` but none of that seems essential.
- Type `python3 -m pip install .`.

Once you've downloaded it, please test it by running the testing
version, at least until I start uploading it to `conda-forge` or
`PyPI` or anywhere else that automatically runs the test suite.

```
python3 -m pip install ".[test]"
pytest
python3 -m pip install .
```

This will also compile the C extension module `siffreadermodule` that does most of the heavy lifting, stick the library into your path for this environment, and then make the SiffPy Python code accessible.

If you want to compile the file reader in debug mode, which logs most of its operations,
please set the `DEBUG` parameter in `setup.py` to `True`. I will make this a command line
option at some point, but for now figuring out the proper way to snag things from the
command line with `distutils`/`setuptools` (instead of just pulling with `sys.argv`)
is a little more work than I'm planning to do.

For now, `siffpy` just does alignment using the intensity images, rather than
considering what source a photon is likely to have arrived from using FLIM.
So it can be run on laptops with relative ease.

### Dependencies 

- If you don't have numpy, this will install it. Uses only basic `numpy` includes, so version won't matter. `numpy` is necessary for `siffreadermodule` to compile, because many of its functions return pointers to `PyArrayObject`s. `scipy` is required for the `registration` submodule, which to me seems like a basic `SiffPy` functionality so I decided to make it a dependency.

## Using SiffPy

The primary object in `SiffPy` is the `SiffReader` class, which is usually imported with `from siffpy import SiffReader` at the start of a notebook. A `SiffReader` object handles I/O with a .siff or .tiff file to keep track of important file-specific variables as well as implements much of the boilerplate sort of code. It's defined in `siffpy.core.SiffReader`.

### Core

This module contains the main `SiffReader` Python object as well
as functionality related to FLIM data (such as multiexponential fitting) and
functionality related to image processing (such as image registration).
TODO: Document me better!

### SiffMath

This submodule analyzes traces, sometimes produced by other modules (like the `SiffTrac` headings) and returns `array`-like objects.
Many plotter classes depend on functionality implemented here, but no functionality here depends on _any_ part of `SiffPy`: it is
_purely_ numerical. This means these functions can be used on any `array`-like classes, and its tools (which return `numpy.ndarray`
subclasses, or `np.ndarray`s themselves) can be used even if all other `SiffPy` functionality's dependencies are not present.

## Handling data

FLIM data is mostly naturally stored in sparse arrays: most pixels are not important, most histogram bins do not have many entries. But once you start building 512 by 512 pixel arrays, each of which have a FLIM measurement depth of 1024 bins, the array sizes get large quickly (one frame of this size would be 536 MB... acquiring at 30 Hz would give you a 16GB array for every one second of imaging). Most of SiffPy's functionality is performed lazily, avoiding loading arrays into memory unless some pixelwise relationship between several arrays is really needed. The `siffreader` C++ module mostly takes a `frames` or `frameslists` argument that allows pooling of frames by index, and the `siffpy` Python API does its best to hide all the nitty-gritty of that process from the user.

## More direct access to the data

The `siffreader` module contains lower-level access to the data, allowing you to directly get numpy arrays from .siff and .tiff files. To learn more, type `import siffreader; help(siffreader)` in your Python interpreter or in a Jupyter notebook.

Note:
So far I've only really been testing functionality in Jupyter notebooks. Note that if you use Jupyter lab, there are a few incompatibilities with the plotting libraries `matplotlib`, `bokeh`, and `holoviews`. However, the core code for extracting the data will be unaffected and relies ONLY on `numpy`.
These issues seem not to be an issue in VSCode's handling of notebooks,
though VSCode can be fussy with `bokeh`.

## Understanding .siff files

.siff files are built to use the skeleton of a .tiff, but instead of each byte (or set of bytes) reflecting a pixel value,
they reflect a photon. As a result, all of the header and IFD structure of a .tiff is present (if you want to know what
those are, feel free to look them up), and so .tiff readers can help you navigate the files if you want to build your
own reader. The structure is as follows (TODO!!!).

### Reading individual frames.
Note that each individual IFD, and corresponding frame, has an additional tag, with the tag ID 907, called SiffCompress.
The SiffCompress flag is one byte, really one bit, reflecting whether that frame uses the compressed siff format or not.
I expect that the future may hold other .siff formats, e.g. sparse count data (the standard .siff tag but without arrival time data, so twice as compact, that lapses into .tiff storage when that becomes more efficient), but probably not up to 255 of them.

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
is stored just before the start of the frame, with essentially what a normal tiff file would contain: a y-size-times-x-size element
array of `uint16`s, with each element corresponding to the number of photons in that pixel. The subsequent block of
bytes is each photon, starting with `y = 0` and increasing `x`, then resetting `x` to 0 and incrementing `y` to 1, and so
forth.

## SiffReaderModule

The `C/C++` code underlying direct interactions with the `.siff` filetype. It produces `numpy.ndarray` objects for image data 
as well as native Python objects for framewise (or experiment-wise) metadata. This mostly is for reading from the file, but
there are a few "smart" things it does, like computing pixel-wise empirical lifetimes given a `FLIMParams` object.