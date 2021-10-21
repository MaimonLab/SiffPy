# SIFFTRAC

The SiffPy interface for FicTrac data that can be aligned to SiffReader data

The central classes are the `FicTracLog` and the `TracPlotter`.

Plotters inherit from the `TracPlotter`, which interfaces with the `FicTracLog`,
to plot and combine different types of data in the `FicTracLog`. Most override
shared methods of the `TracPlotter` class, e.g. `plot`. At least so far, expects @tmohren's 
universal_fictrac_logger framework, so be careful about how your dataframes are parsed! 

## FicTracLog

## TracPlottter

# Individual plotter classes

## TrajectoryPlotter

## HeadingPlotter

Plots the *wrapped* heading using `FicTracLog`'s `integrated_heading_lab` column. The `plot` function takes
the keyword `offset` to offset the heading by a set value