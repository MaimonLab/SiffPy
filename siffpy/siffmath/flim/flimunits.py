"""
For ensuring FlimTraces carrying different types of data are not
accidentally combined on unequal terms
"""

from enum import Enum
from typing import Union

import numpy as np

MULTIHARP_BASE_RESOLUTION_IN_PICOSECONDS = 5

class FlimUnits(Enum):
    PICOSECONDS = "picoseconds"
    NANOSECONDS = "nanoseconds"
    COUNTBINS = "countbins"
    UNKNOWN = "unknown"

def convert_flimunits(in_array : Union[np.ndarray,float], from_units : FlimUnits, to_units : FlimUnits)->Union[np.ndarray,float]:
    """
    Converts an array or float `in_array` from one type of FLIMUnit to another.
    """
    if not (isinstance(from_units, FlimUnits) and isinstance(to_units, FlimUnits)) :
        raise ValueError("Must provide valid FlimUnits to convert")

    if any( unit == FlimUnits.UNKNOWN for unit in [from_units, to_units]) and (not all( unit == FlimUnits.UNKNOWN for unit in [from_units, to_units] )):
        raise ValueError("Unable to convert FlimUnits of UNKNOWN type to any other.")
    
    if from_units is FlimUnits.COUNTBINS:
        if to_units is FlimUnits.PICOSECONDS:
            out = MULTIHARP_BASE_RESOLUTION_IN_PICOSECONDS * in_array
        if to_units is FlimUnits.NANOSECONDS:
            out = (MULTIHARP_BASE_RESOLUTION_IN_PICOSECONDS/1000.0) * in_array
    
    if from_units is FlimUnits.PICOSECONDS:
        if to_units is FlimUnits.COUNTBINS:
            out = in_array/MULTIHARP_BASE_RESOLUTION_IN_PICOSECONDS
        if to_units is FlimUnits.NANOSECONDS:
            out = in_array/1000.0
    
    if from_units is FlimUnits.NANOSECONDS:
        if to_units is FlimUnits.PICOSECONDS:
            out = in_array*1000
        if to_units is FlimUnits.COUNTBINS:
            out = in_array*1000/MULTIHARP_BASE_RESOLUTION_IN_PICOSECONDS