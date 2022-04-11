class NoROIException(Exception):
    """ Raised when a SiffVisualizer or SiffPlotter has no ROIs but ROI methods are called"""
    pass