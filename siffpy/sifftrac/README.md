# SIFFTRAC

The SiffPy interface for FicTrac data that can be aligned to SiffReader data

The central classes are the `FicTracLog` and the `TracPlotter`.

Plotters inherit from the `TracPlotter`, which interfaces with the `FicTracLog`,
to plot and combine different types of data in the `FicTracLog`. Most override
shared methods of the `TracPlotter` class, e.g. `plot`. At least so far, expects @tmohren's 
universal_fictrac_logger framework, so be careful about how your dataframes are parsed! 

Plotters rely on `HoloViews` for data set management and `Bokeh` for plotting. But if your environment
can't import those packages, you can still import this module. You can't use the plotting functions,
and when you import those tools, it will raise `ImportError`s, but you can still use the `FicTracLog`.

## FicTracLog

Parses the `ROS2` node output specified by @tmohen and stores the information as a `pandas` `DataFrame` object.
This is present in an attribute called `dataframe`. But because it's very easy to mess something up because
`DataFrame` objects are mutable, if you attempt to just retrieve this attribute it will instead vocally complain
(print some warning text) and then return a `deepcopy` of the `DataFrame` instead. If you really want to manipulate
the `DataFrame` *by reference* then use the class's function `get_dataframe_reference()`.

Supports alignment to `.siff` imaging data with two functions:

-   `align_to_image_time(siffreader)` : 

        Appends a column to the dataframe that converts epoch time to
        experiment time, as reflected by the frame timestamps in the siff
        or tiff file.

This function is to put everything in a shared timebase relative to the start
of the imaging experiment (rather than the confusing epoch time).

-   `align_to_imaging_data(siffreader, color_channel)` : 

        Appends a column to the dataframe attribute that reflects
        the closest imaging frame, in time, to the FicTrac frame collected.
        Defaults to the lowest color channel

This aligns each row of the `DataFrame` with its closest frame in time. If you
specify a particular color channel, it will return the frames corresponding to
that color channel (they're acquired simultaneously, so the timestamp is the
same regardless of which color you use).

## TracPlotter

`TracPlotter`s and their subclasses implement a `save(path : str = None)`  method that allows you
to pickle each `TracPlotter` so that you can resume analysis later, or at least save the results of
the analyses you performed (most importantly, stored attributes).

# Individual plotter classes

## TrajectoryPlotter

## HeadingPlotter

Plots the *wrapped* heading using `FicTracLog`'s `integrated_heading_lab` column. The `plot` function takes
the keyword `offset` to offset the heading by a set value