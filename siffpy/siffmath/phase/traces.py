import numpy as np
from enum import Enum
from typing import TYPE_CHECKING, Union, Optional, Any
from numpy.typing import NDArray

if TYPE_CHECKING:
    PhaseUnitsLike = Union['PhaseUnits', str]


class PhaseUnits(Enum):
    RADIANS = 'radians'
    DEGREES = 'degrees'


class PhaseTrace(np.ndarray):
    """
    Subclasses the numpy array to create a
    convenient interface for storing information
    about phase. The primary numpy array is the
    estimated phase, but this maintains information
    about, for example: error size, the function used...

    Always bound from -pi to pi on creation -- unless you do
    some numpy operations to it!

    Array is always complex128 -- if the input is NOT
    complex128, it is treated as an array of angles and
    cast to complex with np.exp(1j*input_array).

    __new__ signature is:

    __new__(cls, input_array, method : str = None,
        error_array : np.ndarray = None,
        time : np.ndarray = None,
        info_string : str = '', # new attributes TBD?
        units : PhaseUnitsLike = PhaseUnits('radians'),
    ):

    TODO: MAKE THIS IMPOSE CIRCULAR STATISTICS ON EVERYTHING!
    """
    def __new__(cls,
        input_array : np.ndarray,
        method : Optional[str] = None,
        error_array : Optional[np.ndarray] = None,
        time : Optional[np.ndarray] = None,
        info_string : Optional[str] = '', # new attributes TBD?
        units : 'PhaseUnitsLike' = PhaseUnits('radians'),
        ):
        
        if not (input_array.dtype == np.complex128):
            # Presumes this is an array of angles!
            input_array = np.exp(1j*input_array)

        obj : PhaseTrace = input_array.view(cls)
        
        # add the new attributes to the created instance
        obj.method = method
        obj.error_array = error_array # if 1d, symmetric
        obj.time = time
        obj.units = PhaseUnits(units)
        obj.info_string = info_string

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.method = getattr(obj, 'method', None)
        self.error_array = getattr(obj, 'error_array', None)
        self.time = getattr(obj, 'time', None)
        self.info_string = getattr(obj,'info_string', '')
        self.units = getattr(obj,'units', PhaseUnits('radians'))

    @property
    def angle(self)->NDArray[Any]:
        return np.angle(self, deg = self.units == PhaseUnits('degrees'))

    def convert_units(self, to_units : 'PhaseUnitsLike'):
        to_units = PhaseUnits(to_units)

        if self.units is to_units:
            return
        if self.units == PhaseUnits('radians') and to_units == PhaseUnits('degrees'):
            self[...] *= 180/np.pi
            self.error_array[...] *= 180/np.pi
            self.units = PhaseUnits('degrees')
            return
        if self.units == PhaseUnits('degrees') and to_units == PhaseUnits('radians'):
            self[...] *= np.pi/180
            self.error_array[...] *= np.pi/180
            self.units = PhaseUnits('radians')
            return
