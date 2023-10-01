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

    def conversion_factor(self, other : 'FlimUnits')->float:
        """
        self*conversion_factor = other
        """
        if self is FlimUnits.UNKNOWN or other is FlimUnits.UNKNOWN:
            raise ValueError("Cannot convert from or to UNKNOWN FlimUnits")
        if self is other:
            return 1.0
        if self is FlimUnits.PICOSECONDS:
            if other is FlimUnits.NANOSECONDS:
                return 1/1000.0
            if other is FlimUnits.COUNTBINS:
                return 1.0/BASE_RESOLUTION_PICOSECONDS
        if self is FlimUnits.NANOSECONDS:
            if other is FlimUnits.PICOSECONDS:
                return 1000.0
            if other is FlimUnits.COUNTBINS:
                return 1000.0/BASE_RESOLUTION_PICOSECONDS
        if self is FlimUnits.COUNTBINS:
            if other is FlimUnits.PICOSECONDS:
                return BASE_RESOLUTION_PICOSECONDS
            if other is FlimUnits.NANOSECONDS:
                return BASE_RESOLUTION_PICOSECONDS/1000.0
        raise ValueError("Unable to convert from {} to {}".format(self, other))

FlimUnitsStr = Literal["picoseconds", "nanoseconds", "countbins", "unknown", "unitless"]
FlimUnitsLike = Union['FlimUnits', FlimUnitsStr]

def convert_flimunits(
        in_array : Union[np.ndarray,float],
        from_units : FlimUnitsLike,
        to_units : FlimUnitsLike
    )->Union[np.ndarray,float]:
    """
    Converts an array or float `in_array` from one type of FLIMUnit to another.

    UNITLESS is a special case, where the input is returned unchanged.
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

    conversion_factor = from_units.conversion_factor(to_units)
    return in_array * conversion_factor