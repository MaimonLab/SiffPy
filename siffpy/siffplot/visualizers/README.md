# VISUALIZERS

These classes subclass the `SiffVisualizer`. They handle image streams
and processing that returns other collections of images. They should
default to using `napari` for visualization if applicable, but if possible should
provide `HoloViews` and `Bokeh` based implementations too.