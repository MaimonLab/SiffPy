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
