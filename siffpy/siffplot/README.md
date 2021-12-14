# SIFFPLOT

Plotting interface for `SiffPy`. Uses `HoloViews` and a `Bokeh` backend, and/or `napari` if it can be successfully imported.
`napari` use is restricted for images themselves (e.g. annotating ROIs, visualizing frames) and is optional for all functionality.
`HoloViews` and `Bokeh`, by contrast, are required imports. `napari` seems to be the more pleasant experience, though, and I'll
try my best to maintain full functionality for both, at least as long as I'm the primary programmer on this project (SCT).

If the user has `napari` available, all plotter classes will DEFAULT to `napari`, unless the keyword `use_napari` is set to `False`
on initialization. If `napari` is not available, the keyword is not required -- everything will use `HoloViews` by default.

The primary class for analysis is the `SiffPlotter`,
while the primary class for viewing images is the `SiffVisualizer`. Other classes may inherit from these two, and those classes
can be initialized with any other version of the same parent class to attain their attributes while still performing their
own unique functionality (see below.)

## SiffPlotter

The `SiffPlotter` interfaces with a `SiffReader` that can be applied on initialization. A `SiffPlotter` is initialized with
```
from siffpy.siffplot import SiffPlotter

sp = SiffPlotter(sr)
```

where `sr` is a `SiffReader` object.

The `SiffPlotter` is the main form for interacting with and creating ROIs for analysis. It has a few core methods that will be covered here.
For all other functionality, you can use the documentation for `SiffPlotter` provided by `help(siffpy.siffplot.SiffPlotter)`

### draw_rois

`draw_rois()` returns a `hv.HoloMap` object that can be used to select regions of interest on the reference frames of the associated `siffreader`.
If the `siffreader` does not have any reference frames, i.e. if registration has not been performed, it will throw an error. This is because
there is no "prototypical" frame to use as a reference for drawing ROIs, not because I want to require you to perform registration, so if you'd like
to perform a hasty registration you can just use the average frames for each slice or ScanImage mROI with
`siffreader.registration_dict(reference_method='average')` (including any other `*arg` and `**kwarg` arguments you might want to provide, as
annotated in `siffpy.registration_dict`'s docstring, accessible with `help(siffpy.registration_dict)`).

Once you have reference frames for your `siffreader`, you can simply call `draw_rois()` from the associated `SiffPlotter` and it will return a 
`hv.Layout` of composed reference frames. Each has its own toolbar with `PolyDraw` and `PolyEdit` tool from `Bokeh` that can be used to
select a region or multiple regions of interest. Depend on what types of ROI(s) you want, the type and number of polygons will be different
(see `extract_rois` documentation for more information).

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

## ROIs

`ROI` objects are used by the `siffplot` tools to select, analyze, or emphasize particular components of particular frames of `.siff` data.

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

# DERIVED CLASSES

These are classes which subclass the above few core classes and extend them to allow specialized analyses, plots, etc. without cluttering
up the scope of the primary classes.

## SiffPlotters

### PhasePlotter

The `PhasePlotter` extends the `SiffPlotter` and uses its segmented `ROI` objects to construct estimates of a "phase" of the signal,
which can generally be understood as the complex phase component of the 1D discrete Fourier transform of the segmented ROIs. The
functions of this `SiffPlotter` subclass generally take an `ROI` as an optional argument, and if none is provided will search to
see if they already have an `ROI` attribute. These functions can only apply to `ROI` subclasses with a `subROIs` attribute, which
many classes will link to a custom defined internal class (see, for example, the `Ellipse` class and its `wedges` attribute).
If no such `ROI` is provided or accessible, these methods will raise `Exception`s

## ROIs

Each type of `ROI` is designed for a special anatomical region and contains functions that align with what I expect to be relevant
to those regions in particular. Many contain `subROI` classes, generally used to segment a region (e.g. wedges of the ellipsoid body).
Most ROI classes, as a result, implement a customized `__getattr__` call so that each will return a special internal attribute
when the attribute `subROIs` is requested, while still having an informative name for the attribute in each class (again using the
`Wedge` `subROI` class, the `Ellipse` `ROI` subclass assigns itself an attribute `ellipse.wedges` when the method `segment` is called.
But if you try to get `subROIs` from an `Ellipse`, it will also return the `wedges` attribute). This is so that `subROIs` can be used
as a common interface to all of the functions that might want to use the various types of `subROI` classes.

### Ellipse

The `Ellipse` `ROI` is initialized with a polygon, corresponding to an approximation of its outline. This single polygon will be fit with
an ellipse, or if the argument `center_polygon` is provided (as is an option in the `ellipsoid_body` module's fitting method `fit_ellipse`),
a separate polygon will be fit to the center and the two together will help define the outer ellipse.

An ellipse can be segmented with the method `ellipse.segment(n_segments)`, creating `n_segment` `Wedge` `ROI` objects, stored as a list
in the attribute name `wedges` of the `Ellipse`.

#### Wedges

### Fan



## SiffVisualizers

