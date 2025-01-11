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
        if obj is None:
            return
        self.method = getattr(obj, 'method', None)
        self.error_array = getattr(obj, 'error_array', None)
        self.time = getattr(obj, 'time', None)
        self.info_string = getattr(obj,'info_string', '')
        self.units = getattr(obj,'units', PhaseUnits('radians'))

    @property
    def angle(self)->NDArray[Any]:
        return np.angle(self, deg = self.deg)
    
    @property
    def deg(self)->bool:
        return self.units == PhaseUnits('degrees')

    def convert_units(self, to_units : 'PhaseUnitsLike'):
        """
        Converts the phase trace to the specified units in place

        Example
        -------
        ```python
        import numpy as np
        from siffpy.siffmath.phase.traces import PhaseTrace, PhaseUnits

        theta = np.linspace(0, 2*np.pi, 1000)
        phase_trace = PhaseTrace(np.exp(1j*theta))
        assert all(phase_trace.angle < 10)
        phase_trace.convert_units(PhaseUnits('degrees'))
        assert any(phase_trace.angle > 10)
        phase_trace.convert_units('radians')
        assert all(phase_trace.angle < 10)
        ```
        """

        to_units = PhaseUnits(to_units)

        if self.units is to_units:
            return
        if self.units == PhaseUnits('radians') and to_units == PhaseUnits('degrees'):
            if self.error_array is not None:
                self.error_array[...] *= 180/np.pi
            self.units = PhaseUnits('degrees')
            return
        if self.units == PhaseUnits('degrees') and to_units == PhaseUnits('radians'):
            if self.error_array is not None:
                self.error_array[...] *= np.pi/180
            self.units = PhaseUnits('radians')
            return

    def diff(self)->np.ndarray:
        """
        Returns the angular difference between adjacent
        elements of the phase trace.

        Example
        -------
        ```python
        import numpy as np
        from siffpy.siffmath.phase import PhaseTrace

        theta = np.linspace(0, 2*np.pi, 1000)
        phase_trace = PhaseTrace(np.exp(1j*theta))
        assert np.allclose(np.diff(theta), phase_trace.diff())
        ```
        """
        return np.angle(self[1:]/self[:-1], deg = self.deg)
    
    def abs(self)->np.ndarray:
        """
        Returns the magnitude of the phase trace

        Example
        -------
        ```python
        import numpy as np
        from siffpy.siffmath.phase import PhaseTrace

        theta = np.linspace(0, 2*np.pi, 1000)
        phase_trace = PhaseTrace(np.exp(1j*theta))
        assert np.allclose(1, phase_trace.abs())
        ```
        """
        return np.abs(self)

    def invert(self):
        """
        Reverses the direction of the phase in place -- 
        equivalent to `1.0/self`.

        Example
        -------
        ```python

        import numpy as np
        from siffpy.siffmath.phase import PhaseTrace

        theta = np.linspace(0, 2*np.pi, 1000)
        phase_trace = PhaseTrace(np.exp(1j*theta))
        assert (phase_trace[1:]/phase_trace[:-1]).angle.mean() > 0
        phase_trace.invert() # now goes from 2pi to 0 clockwise
        assert (phase_trace[1:]/phase_trace[:-1]).angle.mean() < 0
        ```
        """
        self[...] = 1.0/self

    def inverted(self)->'PhaseTrace':
        """
        Returns a _new_ PhaseTrace object with the phase inverted without
        modifying the existing trace.

        Example
        -------
        ```python

        import numpy as np
        from siffpy.siffmath.phase import PhaseTrace

        theta = np.linspace(0, 2*np.pi, 1000)
        phase_trace = PhaseTrace(np.exp(1j*theta))
        assert (phase_trace[1:]/phase_trace[:-1]).angle.mean() > 0
        inverted_phase_trace = phase_trace.inverted() # now goes from 2pi to 0 clockwise

        assert (inverted_phase_trace[1:]/inverted_phase_trace[:-1]).angle.mean() < 0
        assert (phase_trace[1:]/phase_trace[:-1]).angle.mean() > 0
        ```
        """
        return 1.0/self