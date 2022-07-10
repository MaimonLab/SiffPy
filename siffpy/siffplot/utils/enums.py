from enum import Enum

class Direction(Enum):
    """ Enums for plotting directions """
    
    HORIZONTAL  =    "horizontal"
    VERTICAL    =    "vertical"

class CorrStyle(Enum):
    """ Enums for style of correlation plots """
    
    LINE        =   "line"
    HEATMAP     =   "heatmap"