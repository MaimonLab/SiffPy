# SIFFPLOT

Plotting interface for `SiffPy`. Uses `HoloViews` and a `Bokeh` backend, and/or `napari` if it can be successfully imported.
`napari` use is restricted for images themselves (e.g. annotating ROIs, visualizing frames) and is optional for all functionality.
`HoloViews` and `Bokeh`, by contrast, are required imports. `napari` seems to be the more pleasant experience, though, and I'll
try my best to maintain full functionality for both, at least as long as I'm the primary programmer on this project (SCT).

Almost none of this relies on any serious `SiffReader` functionality, and as a result many of the classes can actually be used
for .tiff files read by a `SiffReader`, or even previously processed `numpy` arrays. TODO: `DummySiffReader` to allow the deception.

If the user has `napari` available, all `SiffVisualizer` classes will DEFAULT to `napari`, unless initialized with the keyword
argument `backend = 'HoloViews'`. 
If `napari` is not available, the keyword is not required -- everything will use `HoloViews` by default. To use
options, and to actually visualize the `HoloViews` objects, be sure to call `hv.extension('bokeh')` in a notebook or script. Just
as a warning, at the time of writing (SCT Dec. 28 2021), the `extension` call doesn't play nicely with `napari` and Notebooks, and
it makes you have to manually execute individual cells, which is kind of a pain.

The primary class for analysis is the `SiffPlotter`,
while the primary class for viewing image streams is the `SiffVisualizer`.
Other classes may inherit from these two, and those classes
can be initialized with any other version of the same parent
class to attain their attributes while still performing their
own unique functionality (see below). The `SiffVisualizer` deals
with image visualization directly, while the `SiffPlotter` takes fluorescence / image data
and produces data-type plots (heatmaps, traces, etc.).

`ROI` subclasses will also do everything entirely in `HoloViews`, but I intend to implement functionality
to get them to be visualizable in at least some types of `SiffVisualizer`s.

## SiffPlotter

The `SiffPlotter` interfaces with a `SiffReader` that can be applied on initialization. A `SiffPlotter` is initialized with
```
from siffpy.siffplot import SiffPlotter

sp = SiffPlotter(sr)
```

where `sr` is a `SiffReader` object.

The `SiffPlotter` is the main form for interacting with and creating ROIs for analysis. It has a few core methods that will be covered here.
For all other functionality, you can use the documentation for `SiffPlotter` provided by `help(siffpy.siffplot.SiffPlotter)`.

All `SiffPlotter` subclasses implement a `visualize(*args, **kwargs)` method that plots the 'standard output' of computations associated
with that subclass of `SiffPlotter`.

### visualize

`visualize(*args, **kwargs)` is usually implemented as a custom method for each subclass of the `SiffPlotter`. It either returns
a `HoloViews.Layout` object or it modifies its active `viewer` attribute (if it's using `holoviews` or `napari` as a backend,
respectively). If the selected backend is not compatible with the `SiffPlotter` subclass's `visualize` method, it will raise
an `AttributeError` rather than just plotting it anyways.

## ROIs

`ROI` objects are used by the `siffplot` tools to select, analyze, or emphasize particular components of particular frames of `.siff` 
data. More data can be found below and in the README inside the `roi_protocols` directory. To learn more about available fitting
methods, call `siffpy.siffplot.ROI_extraction_methods()`.

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


### draw_rois

`draw_rois()` functionality, regardless of backend, relies on using the reference frames of the associated `siffreader`.
If the `siffreader` does not have any reference frames, i.e. if registration has not been performed, it will throw an error. This is because
there is no "prototypical" frame to use as a reference for drawing ROIs, not because I want to require you to perform registration, so if you'd like
to perform a hasty registration you can just use the average frames for each slice or ScanImage mROI with
`siffreader.registration_dict(reference_method='average')` (including any other `*arg` and `**kwarg` arguments you might want to provide, as
annotated in `siffpy.registration_dict`'s docstring, accessible with `help(siffpy.registration_dict)`).

__(`HoloViews` version)__

`draw_rois()` returns a `hv.HoloMap` object that can be used to select regions of interest on the reference frames of the associated `siffreader`.
The `hv.HoloMap` is stored in the attribute `annotation_dict` because it uses `HoloViews` `annotator` objects.

__(`napari` version)__

`draw_rois()` creates a `napari.Viewer` object that can be used to draw polygons, lines, and other shapes on the `siffreader`'s reference frames.
Shapes are drawn on a `napari.shapes.shapes.Shapes` layer with the name `'ROI shapes'` and can be accessed with the `SiffPlotter`'s `viewer` attribute's
`layers` attribute, a list of the the layers of the `viewer`.

### extract_rois

`extract_rois(region_name, method_name, *args, **kwargs)` returns `None` but creates a list of `siffpy.siffplot.roi_protocols.rois.ROI` elements
(or, really, subclasses of such) and stores them in the `SiffPlotter` attribute `rois`. `region_name` and `method_name` are both strings that are
obligate arguments, though `None` may be provided for `method_name` and the default for `region_name` will be used. Region names, and their
corresponding available methods, can be found with `siffpy.siffplot.ROI_extraction_methods()`, which prints a string documenting each
possible region and their corresponding methods. Every method takes the `annotation_dict` from `draw_rois` and the `reference_frames` from
the `siffreader` as obligate arguments, but that's handling within the `extract_rois` code. More interestingly, each method will typically
also permit keyword arguments bespoke to that particular method. Those can be provided to `extract_rois` as `**kwargs`, which are all passed
directly to the method. After the polygons highlighted in `draw_rois` have been used to create an ROI or ROI objects, they can be accessed in
the `SiffPlotter` attribute `rois`, e.g. to pass to `select_points`.

### select_points

`select_points(roi)` returns a `hv.Overlay` that show the main polygon object of a `siffpy.siffplot.roi_protocols.rois.ROI` class, overlaid
on its reference frame (if available), and allows selection of points on the polygon. A single click will add the nearest vertex of the polygon, and
a double click will remove the nearest selected vertex of the polygon. These points are associated with that ROI and will be stored when the ROI
is saved, and can be used for subsequent processing.

# DERIVED CLASSES

These are classes which subclass the above few core classes and extend them to allow specialized analyses, plots, etc. without cluttering
up the scope of the primary classes.

## SiffPlotters

Different `SiffPlotter` subclasses have details that may be elaborated on in TODO: Find a place for this!

### FluorescencePlotter

### PhasePlotter

The `PhasePlotter` extends the `FluorescencePlotter` and uses its segmented `ROI` objects to construct estimates of a "phase" of the signal,
which can generally be understood as the complex phase component of the 1D discrete Fourier transform of the segmented ROIs. The
functions of this `SiffPlotter` subclass generally take an `ROI` as an optional argument, and if none is provided will search to
see if they already have an `ROI` attribute. These functions can only apply to `ROI` subclasses with a `subROIs` attribute, which
many classes will link to a custom defined internal class (see, for example, the `Ellipse` class and its `wedges` attribute).
If no such `ROI` is provided or accessible, these methods will raise `Exception`s

### HistogramPlotter

This plotter deals specifically with arrival time histograms in `.siff` data.

### RegistrationPlotter

This plotter is for tracking details of the registration methods' outputs.

## ROIs

Each type of `ROI` is designed for a special anatomical region and contains functions that align with what I expect to be relevant
to those regions in particular. Many contain `subROI` classes, generally used to segment a region (e.g. wedges of the ellipsoid body).
Most ROI classes, as a result, implement a customized `__getattr__` call so that each will return a special internal attribute
when the attribute `subROIs` is requested, while still having an informative name for the attribute in each class (again using the
`Wedge` `subROI` class, the `Ellipse` `ROI` subclass assigns itself an attribute `ellipse.wedges` when the method `segment` is called.
But if you try to get `subROIs` from an `Ellipse`, it will also return the `wedges` attribute). This is so that `subROI`s can be used
as a common interface to all of the functions that might want to use the various types of `subROI` classes. More on `ROI`s, `subROI`s, and all other things `ROI` related in the `roi_protocols` README file.

## SiffVisualizers

