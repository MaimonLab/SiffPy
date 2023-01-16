import numpy as np
from enum import Enum

class PhaseUnits(Enum):
    pass


class PhaseTrace(np.ndarray):
    """
    Subclasses the numpy array to create a
    convenient interface for storing information
    about phase. The primary numpy array is the
    estimated phase, but this maintains information
    about, for example: error size, the function used...

    Always bound from 0 to 2pi on creation -- unless you do
    some numpy operations to it!

    TODO: USE COMPLEX NUMBERS INSTEAD OF RAW ANGLES

    __new__ signature is:

    __new__(cls, input_array, method : str = None,
        error_array : np.ndarray = None,
        time : np.ndarray = None,
        info_string : str = '', # new attributes TBD?
    ):

    TODO: MAKE THIS IMPOSE CIRCULAR STATISTICS ON EVERYTHING!
    """
    def __new__(cls, input_array, method : str = None,
        error_array : np.ndarray = None,
        time : np.ndarray = None,
        info_string : str = '', # new attributes TBD?
        ):
        
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        arr = np.asarray(input_array)
        arr = arr % (2.0*np.pi)
        obj : PhaseTrace = arr.view(cls)
        
        # add the new attributes to the created instance
        obj.method = method
        obj.error_array = error_array # lower bound, upper bound
        obj.time = time
        obj.units = 'radians'
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