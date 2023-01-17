# SiffReaderModule

C and C++ code compiled into a local Python extension module. The code in here is designed to quickly read .siff files and .tiff files (with a custom tiff-like reader) and return Numpy arrays. I'll document this too, though it's a bit messier.

TODO:
- The `SiffReader` object has gotten extremely complex, and the way
some function calls are handled is not great (and some things maybe should be inlined that aren't).

- Clean up memory leaks in the `registrationDict` argument. There are a few kB of memory leaks when
using functions that have a `registrationDict` keyword argument IF you don't pass an argument.
No leaks if you always provide one (`PyTuple_Pack` and `_SetItem` steal references).
It's because these functions make a registrationDict of all 0s (which
is silly, since I define a version of the functions that don't use a registrationDict too...).
TO FIX THIS: Write a version of the `siffreader.cpp` functions that do not use a `registrationDict`
argument, then call those if `registrationDict == NULL`.

- Implement a debug logger file so that as this toolkit grows it's easy to follow errors and crashes.

## SiffIO object

As of `siffpy` version `0.6.0`, the preferred way to access
siff file inputs and outputs is to create a `SiffIO` object

```
from siffreadermodule import SiffIO

siffio = SiffIO() # filepath may be provided as a string in initialization

siffio.open(filepath) # if a SiffIO was initialized without a file

siffio.get_frames(*args, **kwargs)

siffio.sum_roi_flim(*args, **kwargs)
```

which is what the `siffpy.SiffReader` object does as of
`0.6.0`.


## SiffReader class (C++)

Most file I/O is done by a `SiffReader` object in the module (yes, I agree, it was maybe a bad idea to name both the module and the `C++` class `SiffReader`, though the idea is that the `Python` module wraps the `C++` class).

The `SiffReader` has an `ifstream` object that is opened at the start of a session and which handles all reading of the
file, `SiffReader.siff`. The front-facing methods can:

- `open` a file (with a provided `std::string` filename)
- `retrieveFrames` with an array of `uint64_t` frame numbers from an open file
- `poolFrames` with a `PyObject*` to a `list` object, each element of which is itself a list of frame numbers
- `flimMap`, which takes a `PyObject*` to a `siffpy.FLIMParams` object and a second `PyObject*` to lists of lists of frames, as with `pool_frames`
- `readMetaData` to get frame-leading specific data (specified as an array of `uint64_t`).
- `readFixedData` to get metadata that is established at the start of an experiment.
- `siffToTiff`, saves a `.tiff` file that contains the intensity data of a `.siff` file.

## SiffIO methods (Python module)

You can interact with the `SiffReader` class through the `SiffIO` Python class.

For more information, and probably more up to date than the README, try 
```
from siffreadermodule import SiffIO

help(SiffIO)
```

The methods are (at time of writing):

-    `SiffIO.open(filename : str) -> None`
        Opens the file at location filename.

-    `SiffIO.close() -> None`
        Closes an open file.

-    `SiffIO.get_file_header()->dict` 
        Returns header data that applies across the file. Dictionary returned contains the following `(key, value)` pairs:
        - `Filename` : `str`
                The path to the file
        - `BigTiff`  : `bool`
                Whether the file is a "BigTiff" (uses 64-bit pointers)
        - `IsSiff`   : `bool`
                Whether the file is a "Siff" (SImplified Flim Format)
        - `Number of frames` : `int` (`Py_ssize_t`)
                Total number of frames in the file (that's all colors, time points, z slices...)
        - `Non-varying frame data` : `str`
                A long string containing the output from `ScanImage` stored at the start of an experiment. This
                includes many experimental parameters, as well as acquisition parameters and so on.
        - `ROI string` : `str`
                A string containing the information about `mROI` data output by `ScanImage` at the start of an experiment.
        - `IFD pointers` (only in debug mode) : `list[int]`
                A list of pointers to the start of the IFD for every frame, to jump directly to that frame for file I/O

-    `SiffIO.get_frames(frames= None, type=list, flim=False, registration = None, discard_bins = None)->list[np.ndarray]`
        Returns frames as a list of numpy arrays.

-    `SiffIO.get_frame_metadata(frames=[])->list[dict]`
        Returns frame metadata in the form of a list of dicts of the same size as `frames`. The dictionaries are of the form `(key, value)`:
        - `Width` : `int`
            The x dimension of the frame
        - `Length` : `int`
            The y dimension of the frame
        - `endOfIFD` : `int`
            A pointer to the end of the IFD for this frame (the beginning of the meta data).
        - `dataStripAddress` : `int`
            A pointer to the beginning of the sequential data, the pixel-wise values of the frame.
        - `stringLength`    : `int`
            The length of the string containing metadata about the frame.
        - `X Resolution` : `int`
            A rational describing the size of an individual pixel (TODO come back to explanation)
        - `YResolution` : `int`
            A rational describing the size of an individual pixel (TODO come back to explanation)
        - `Bytecount` : `int`
            The length of the string of sequential data (proportional to the number of photons for a `.siff` and the
            number of pixels for a `.tiff`).
        - `Frame metadata` : `str`
            A string representation of the metadata for this frame (e.g. timestamps, appended text).
        -  `Siff compression` : `bool`
            Whether or not the frame uses the siff compression algorithm for data (described below).

-    `SiffIO.pool_frames(pool_lists : list[list[int]], type=list, flim=False, registration=None)-> list[np.ndarray]`
        As `get_frames`, but takes a list of lists as input. Each top-level list element contains a list of
        frame numbers, which themselves will be pooled together to return a single `np.ndarray` that is the
        summed values of the photon counts of the frames specified by that list.

-    `SiffIO.flim_map(params : siffpy.FLIMParams, framelist = None, confidence_metric= 'log_p', registration=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
        Returns a tuple: empirical lifetime, intensity, and a confidence metric. `framelist` is as in `pool_frames`.
        TODO: document this more (explain the confidence metric, etc.)

-    `SiffIO.get_histogram(frames=None)->np.ndarray`
        Returns histogrm of photon arrival times.

-    `SiffIO.num_frames() -> int`
        If file is open, reports the total number of frames (quick access to a useful field of the `SiffReader`)

There are also `siffreadermodule` methods:

-    `suppress_warnings() -> None`
        Suppresses module-specific warnings.

-    `report_warnings() -> None`
        Allows module-specific warnings (undoes suppress_warnings)

-    `debug() -> None`
        Enables siffreadermodule debugging log and reporting

-    `siff_to_tiff(sourcepath : str, savepath : str = None) -> None`
        Converts the open .siff file to a .tiff file, discarding arrival time information, if relevant. If no path is
        provided, the .tiff is saved in the same location as the .siff.