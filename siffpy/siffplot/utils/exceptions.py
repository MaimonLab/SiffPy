from enum import Enum

class NoROIException(AttributeError):
    """ Raised when a SiffVisualizer or SiffPlotter has no ROIs but ROI methods are called"""
    pass

class StyleError(ValueError):
    """ Raised when an invalid style parameter is passed to any plotters with style interfaces"""

    def __init__(self, style_enum_class : type):
        super().__init__(
            f"Invalid style. Try any of {[x[1].value for x in style_enum_class.__members__.items()]}"
        )