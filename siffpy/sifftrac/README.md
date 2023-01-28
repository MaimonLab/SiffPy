# SIFFTRAC

The SiffPy interface for FicTrac data that can be aligned to siff data

The central class is the `FicTracLog`.

- TODO: POLARS!

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