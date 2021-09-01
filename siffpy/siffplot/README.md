# SIFFPLOT

Plotting interface for `SiffPy`. Uses `HoloViews` and a `Bokeh` backend. The primary class is the `SiffPlotter`

## SiffPlotter

The `SiffPlotter` interfaces with a `SiffReader` that can be applied on initialization. A `SiffPlotter` is initialized with

`from siffpy.siffplot import SiffPlotter`
`sp = SiffPlotter(sr)`

where `sr` is a `SiffReader` object