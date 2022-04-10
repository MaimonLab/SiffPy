class NoROIException(Exception):
    """ Raised when a SiffVisualizer has no ROIs but ROI methods are called"""
    pass