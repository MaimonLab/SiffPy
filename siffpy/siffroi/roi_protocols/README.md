# ROI PROTOCOLS

This section contains code pertaining to annotation of, segmentation of,
and general analysis of ROIs in image data. All code in this section is
actually independent of any `siffpy` functionality, and so can be used
on any `numpy` or `numpy`-like data, in principle. These methods produce
objects that subclass the `ROI` class. The `ROI` superclass contains methods
that are common to all `ROI` tools (e.g. `mask` to return a `numpy.ndarray`),
while each subclass TODO FINISH

`ROI` classes are created using `ROIProtocol` subclasses.

## ROIProtocol

This contains the basic interface for segmenting out ROIs of interest.
An `ROIProtocol` implements at least one method `extract(self, *args, **kwargs)->ROI`. It has additional parameters that can often shape how this extraction
is done, and the `inspect` module is used to parse those parameters and
make them accessible via GUI. Each individual `ROIProtocol` subclass is
defined within its respective modules (e.g. `ellipsoid_body`).

## ROI

## ROI Subclasses

### Ellipse

The `Ellipse` `ROI` is initialized with a polygon, corresponding to an approximation of its outline. This single polygon will be fit with
an ellipse, or if the argument `center_polygon` is provided (as is an option in the `ellipsoid_body` module's fitting method `fit_ellipse`),
a separate polygon will be fit to the center and the two together will help define the outer ellipse.

An ellipse can be segmented with the method `ellipse.segment(n_segments)`, creating `n_segment` `Wedge` `ROI` objects, stored as a list
in the attribute name `wedges` of the `Ellipse`.

### Fan

## SubROIs

The `subROI` is implemented for each `ROI` subclass, and corresponds to a special type of `ROI` used to segment larger `ROI`
objects into smaller chunks. The method of segmentation is customized for each `subROI` class and `ROI` subclass, but they
share some common attributes. For one, they themselves are `ROI`s and so they have every method an `ROI` in general does.
TODO finish!

### Wedges

### Columns

## Methods

This submodule also contains many methods for fitting `ROI` classes in general. Each region has its own Python file
`<region>.py` containing methods pertaining to fitting ROIs to that part of the fly brain. These methods can all be
invoked by the method `siffpy.siffroi.roi_protocols.roi_protocol(region, method_name, *args, **kwargs)` which takes
two strings, one for the region (which can have many allowable aliases) and one for the method method name, along with
any of those methods' required arguments or optional keyword arguments. Available methods can be found by calling the
method `siffpy.siffplot.ROI_extraction_methods()` which both prints a `str` and returns a dictionary of
`str`s for each region containing the function signatures and docstrings of all regions' ROI methods. The dictionary
keys are organized by region and contain specifically the docstrings for functions for that region as values.

## Dependencies

All of these methods rely on `HoloViews`, and can be visualized either
using `Bokeh` or `napari` (which, at the moment, are slightly incompatible
in interactive mode in Jupyter notebooks, but otherwise can play nice with
one another). By default, all methods remain agnostic about visualization
purposes, except for once `visualize` is called by an `ROI` class, which
relies on initialization of `HoloViews` plotting (and thus its injection
into the IPython event loop).