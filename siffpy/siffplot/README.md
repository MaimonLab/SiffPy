# SIFFPLOT

Plotting interface for `SiffPy`. Uses `HoloViews` and a `Bokeh` backend. The primary class for analysis is the `SiffPlotter`,
while the primary class for viewing images is the `SiffVisualizer`.

## SiffPlotter

The `SiffPlotter` interfaces with a `SiffReader` that can be applied on initialization. A `SiffPlotter` is initialized with
```
from siffpy.siffplot import SiffPlotter

sp = SiffPlotter(sr)
```

where `sr` is a `SiffReader` object.

## SiffVisualizer

The `SiffVisualizer` is used to show frames read by a `SiffReader`. It also requires a `SiffReader` for initialization as follows:
```
from siffpy import SiffReader
from siffpy.siffplot import SiffVisualizer

path_to_file : str = "path_to_file"

sr = SiffReader()
sr.open(path_to_file) # Not actually required, but view_frames will throw an error if the SiffReader does not have an open file

sv = SiffVisualizer(sr)
sv.view_frames()
```

`view_frames` *dynamically* reads from the `SiffReader`, so it occupies very little RAM. However, it requires you to retain access
to the file (so it's annoying if you're trying to read from a server) and, because it's making use of the `SiffReader` Python object's
underlying `C++` `SiffReader` class, which uses a single `std::istream` object (its `siff` member) to read the file, might not execute while you're running
other lines of code that themselves are using the `std::istream`. So an alternative usage is to load all of the frames into memory at once:

```
sv.view_frames(load_frames = True)
```

which will take much longer up front but resolves the other issues. However, this means you won't be able to adjust the `pool_width` parameter,
which adjusts how many successive frames are merged.
TODO: Update this to enable flexible `pool_frames` in loaded frames

