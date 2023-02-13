# CORE

Functionality for the main `SiffReader` class. 

TODO:

-   Documentation here
-   Spread `SiffReader` functionality over more files
rather than having one long file.
-   Improve the `tqdm` interface so it doesn't do the
hacky `__IPYTHON__` trick

## ImParams

Microscopy experiments often involve more than just a sequence of frames.
One example is using a piezo to control the z position of the objective
very quickly. Still, the piezo is often not quite fast enough, and certain
frames are deliberately discarded as 'flyback' frames. So when using
functionality to get frames, or data from frames, using `SiffReader`
methods, it's hard to know beforehand which frames to get. For this reason,
each `SiffReader` object has an attribute `im_params` storing an `ImParams`
object. The `ImParams` object knows everything about the acquisition parameters
of a given experiment, and has functions to return lists of frames (and,
in the process, discard irrelevant ones), as well as providing other useful
imaging details (`picoseconds_per_bin`, for example, with FLIM data).

The `ImParams` object has several frame methods:

TODO: make `flatten` a keyword argument for these, so that there
aren't as many redundant methods

- ` framelist_by_timepoint` : returns a `list[list[int]]`, with
the largest scope list corresponding to each timepoint, and each
inner list corresponding to the indices of the frames for that
timepoint
- `framelist_by_color`: returns a `list[int]` with the indices
of all frames in the color channel passed in.
- `framelist_by_slice`: returns a list of all frames corresponding
to the `z` plane and `color_channel` passed in.
- `framelist_by_slices`: returns a list of all frames corresponding to
the frames of _all_ `z` planes passed in, flattened (maybe should be
merged into `framelist_by_slice`).
- `flatten_by_timepoints`: returns a `list[int]` of all frames
within the requested timepoints for a given slice (or all slices).