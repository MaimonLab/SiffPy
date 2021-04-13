import holoviews as hv
import bokeh
from holoviews import opts
hv.extension('bokeh')

class SiffPlot(hv.DynamicMap):
    """
    Extends the Holoviews DynamicMap class
    to facilitate doing analyses that are
    more convenient for my data.
    """

    def __init__(self, *args, **kwargs):
        super(SiffPlot, self).__init__(*args, **kwargs) # call the DynamicMap init


