"""
For ensuring FlimTraces carrying different types of data are not
accidentally combined on unequal terms
"""

from enum import Enum
from typing import Union, Literal

import numpy as np

MULTIHARP_BASE_RESOLUTION_IN_PICOSECONDS = 5

## THIS IS BAD!
# TRYING TO COME UP WITH A BETTER WAY
# TO USE THE READ INFORMATION FROM THE
# FILE
BASE_RESOLUTION_PICOSECONDS = 4*MULTIHARP_BASE_RESOLUTION_IN_PICOSECONDS

class FlimUnits(Enum):
    PICOSECONDS = "picoseconds"
    NANOSECONDS = "nanoseconds"
    COUNTBINS = "countbins"
    UNKNOWN = "unknown"
    UNITLESS    = "unitless"

FlimUnitsStr = Literal["picoseconds", "nanoseconds", "countbins", "unknown", "unitless"]
FlimUnitsLike = Union['FlimUnits', FlimUnitsStr]

def convert_flimunits(
        in_array : Union[np.ndarray,float],
        from_units : FlimUnitsLike,
        to_units : FlimUnitsLike
    )->Union[np.ndarray,float]:
    """
    Converts an array or float `in_array` from one type of FLIMUnit to another.

    UNITLESS nondimensionalizes assuming a 12.5 nanosecond unit, i.e. 80 MHz, approximately
    the longest timescale one would expect under usual conditions
    """
    if isinstance(from_units, str):
        from_units = FlimUnits(from_units)
    if isinstance(to_units, str):
        to_units = FlimUnits(to_units)

    if from_units == to_units:
        return in_array

    if not (isinstance(from_units, FlimUnits) and isinstance(to_units, FlimUnits)) :
        raise ValueError("Must provide valid FlimUnits to convert")

    if from_units is FlimUnits.UNITLESS:
        return in_array

    if any( unit == FlimUnits.UNKNOWN for unit in [from_units, to_units]) and (not all( unit == FlimUnits.UNKNOWN for unit in [from_units, to_units] )):
        raise ValueError("Unable to convert FlimUnits of UNKNOWN type to any other.")
    
    if from_units is FlimUnits.COUNTBINS:
        if to_units is FlimUnits.PICOSECONDS:
            out = BASE_RESOLUTION_PICOSECONDS * in_array
        if to_units is FlimUnits.NANOSECONDS:
            out = (BASE_RESOLUTION_PICOSECONDS/1000.0) * in_array
        if to_units is FlimUnits.UNITLESS:
            out = (BASE_RESOLUTION_PICOSECONDS/12500) * in_array
    
    if from_units is FlimUnits.PICOSECONDS:
        if to_units is FlimUnits.COUNTBINS:
            out = in_array/BASE_RESOLUTION_PICOSECONDS
        if to_units is FlimUnits.NANOSECONDS:
            out = in_array/1000.0
        if to_units is FlimUnits.UNITLESS:
            out = in_array/12500
    
    if from_units is FlimUnits.NANOSECONDS:
        if to_units is FlimUnits.PICOSECONDS:
            out = in_array*1000
        if to_units is FlimUnits.COUNTBINS:
            out = in_array*1000/BASE_RESOLUTION_PICOSECONDS
        if to_units is FlimUnits.UNITLESS:
            out = in_array/12.5

    if from_units is FlimUnits.UNITLESS:
        if to_units is FlimUnits.PICOSECONDS:
            out = in_array*12500
        if to_units is FlimUnits.NANOSECONDS:
            out = in_array*12.5
        if to_units is FlimUnits.COUNTBINS:
            out = in_array*12500/BASE_RESOLUTION_PICOSECONDS
    return out