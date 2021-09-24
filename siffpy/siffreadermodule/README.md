# SiffReaderModule

C and C++ code compiled into a local Python extension module. The code in here is designed to quickly read .siff files and .tiff files (with a custom tiff-like reader) and return Numpy arrays. I'll document this too, though it's a bit messier.

## SiffReader class (C++)

Most file I/O is done by a `SiffReader` object in the module (yes, I agree, it was maybe a bad idea to name both the `PyObject` and the `C++` class `SiffReader`, though the idea is that the `Python` class wraps the `C++` class).

The `SiffReader` has an `ifstream` object that is opened at the start of a session and which handles all reading of the
file, `SiffReader.siff`. The front-facing methods can:

- `open` a file (with a provided `std::string` filename)
- `retrieveFrames` with an array of `uint64_t` frame numbers from an open file
- `poolFrames` with a `PyObject*` to a `list` object, each element of which is itself a list of frame numbers
- `flimMap`, which takes a `PyObject*` to a `siffpy.FLIMParams` object and a second `PyObject*` to lists of lists of frames, as with `pool_frames`
- `readMetaData` to get frame-leading specific data (specified as an array of `uint64_t`).
- `readFixedData` to get metadata that is established at the start of an experiment.
- `siffToTiff`, saves a `.tiff` file that contains the intensity data of a `.siff` file.

## SiffReader methods (Python module)

You can interact with the `SiffReader` class through the `siffreader` Python module. At import, this module initializes a `SiffReader` which you then can interact with. __That means that there's a single global (static) `SiffReader` within the interpreter__, and so it has memory of function calls made previously (e.g. to change settings, or which file is being read). 

For more information, and probably more up to date than the README, try 
```
import siffreader

help(siffreader)
```

The methods are (at time of writing):

-    `open(filename : str) -> None`
        Opens the file at location filename.

-    `close() -> None`
        Closes an open file.

-    `get_file_header()->dict` 
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

-    `get_frames(frames= None, type=list, flim=False, registration = None, discard_bins = None)->list[np.ndarray]`
        Returns frames as a list of numpy arrays.

-    `get_frame_metadata(frames=[])->list[dict]`
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

-    `pool_frames(pool_lists : list[list[int]], type=list, flim=False, registration=None)-> list[np.ndarray]`
        As `get_frames`, but takes a list of lists as input. Each top-level list element contains a list of
        frame numbers, which themselves will be pooled together to return a single `np.ndarray` that is the
        summed values of the photon counts of the frames specified by that list.

-    `flim_map(params : siffpy.FLIMParams, framelist = None, confidence_metric= 'log_p', registration=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
        Returns a tuple: empirical lifetime, intensity, and a confidence metric. `framelist` is as in `pool_frames`.
        TODO: document this more (explain the confidence metric, etc.)

-    `get_histogram(frames=None)->np.ndarray`
        Returns histogrm of photon arrival times.

-    `suppress_warnings() -> None`
        Suppresses module-specific warnings.

-    `report_warnings() -> None`
        Allows module-specific warnings (undoes suppress_warnings)

-    `num_frames() -> int`
        If file is open, reports the total number of frames (quick access to a useful field of the `SiffReader`)

-    `debug() -> None`
        Enables siffreadermodule debugging log and reporting

-    `sifftotiff(savepath : str = None) -> None`
        Converts the open .siff file to a .tiff file, discarding arrival time information, if relevant. If no path is
        provided, the .tiff is saved in the same location as the .siff.

