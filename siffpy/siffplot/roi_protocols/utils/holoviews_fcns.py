from siffpy.siffplot.roi_protocols.utils.polygon_sources import PolygonSource, VizBackend


class PolygonSourceHoloviews(PolygonSource):
    def __init__(self, holoviews_source : object):
        super().__init__(VizBackend.HOLOVIEWS, holoviews_source)

    @property
    def polygons(self):
        return self.source

    def get_largest_polygon(self, slice_idx = None, n_polygons = 1):
        pass

    def get_largest_lines(self, slice_idx = None, n_lines = 2):
        pass

    def get_largest_ellipse(self, slice_idx = None, n_ellipses = 1):
        pass