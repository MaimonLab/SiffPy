.. _multi_roi:

Reading and analyzing data from Multi-ROI imaging
====================================================

``ScanImage`` offers Multi-ROI imaging, which allows multiple
fields of view that are imaged in sequence instead of single block
frames. These are stored in ``.siff`` files as all concatenated in
one frame, typically with blank lines in between (at least in resonant
scanning mode). To read these data, you can either read the whole frame,
for example with ``siffreader.get_frame(...)`` or you can use the ``mROI``
tools. These are a combination of attributes of the ``ImParams`` class and
``SiffReader`` methods that 1) parse the metadata to determine how to extract
each of the ROIs and 2) extract the ROIs as flat masks (since they may be of
arbitrary, not even rectangular, shapes). They also contain tools to cast these
masks back to otherwise-``nan`` arrays of the same shape as the full frame.

Example code
----------------


