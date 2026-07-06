"""
Got tired of all the various timeseries types I'm using and
their confusing units. This is a simply numpy array extension
that simply adds units, a few extra fields, and provides
methods to interconvert by units
"""
from typing import Any, Generic, Literal, TypeVar, Union
from enum import Enum

import numpy as np

TimestampLike = Union[int, float, 'np.dtype[np.int_]', 'np.dtype[np.float64]']

class TimeseriesUnits(Enum):
    """
    Enum for timeseries units
    """
    experiment_seconds = 'experiment_seconds'
    epoch_nanoseconds = 'epoch_nanoseconds'
    epoch_seconds = 'epoch_seconds'

# A number that can be cast to a TimeseriesUnits
TimeUnitsLike = Union[TimeseriesUnits, Literal['experiment_seconds', 'epoch_nanoseconds', 'epoch_seconds']]

T = TypeVar('T', bound=TimeUnitsLike)
A = TypeVar('A', bound=np.ndarray[Any, Union[np.dtype[np.floating], np.dtype[np.uint64]]])

def convert_from_unit_to_unit(
    from_units : TimeUnitsLike,
    to_units : TimeUnitsLike,
    value : A,
) -> A:
    """
    Takes an array in one unit (from_units)
    and converts it to another (to_units).

    Arguments
    ---------

    from_units : TimeUnitsLike
        The units of the input array

    to_units : TimeUnitsLike
        The units of the output array

    value : np.ndarray
        The array to convert
    """
    from_units = TimeseriesUnits(from_units)
    to_units = TimeseriesUnits(to_units)

    if from_units == to_units:
        return value
    
    if from_units == TimeseriesUnits.epoch_nanoseconds:
        if to_units == TimeseriesUnits.epoch_seconds:
            return value / 1e9
        else:
            raise ValueError(f'Unknown to_units {to_units}')
        
    if from_units == TimeseriesUnits.epoch_seconds:
        if to_units == TimeseriesUnits.epoch_nanoseconds:
            return value * 1e9
        else:
            raise ValueError(f'Unknown to_units {to_units}')

    raise NotImplementedError('Not implemented yet')
    
    #if from_units == TimeseriesUnits.experiment_seconds:
    

class Timeseries(np.ndarray, Generic[T]):
    """
    Extends the numpy array to contain
    unitful attributes and get mad if you
    try to combine two Timeseries objects
    with incompatible units (or automatically converts
    them). Todo : sometimes a timeseries has its own reference point,
    `time_zero`, that can be used to convert to/from epoch time. This
    should be allowable here, but I haven't decided on a 'right' way
    yet.
    """

    def __new__(
        cls,
        input_array : np.ndarray,
        units : T = TimeseriesUnits.experiment_seconds,
        #time_zero : TimestampLike = 0,
    ):
        """
        Create a new Timeseries object
        """
        obj = np.asarray(input_array).view(cls)
        obj.units = TimeseriesUnits(units)
        #obj.time_zero = time_zero
        return obj
    
    def __array_finalize__(self, obj):
        """
        Finalize the array
        """
        if obj is None:
            return
        self.units = getattr(obj, 'units', None)
        #self.time_zero = getattr(obj, 'time_zero', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override the ufunc method to check units
        """
        # Check units
        for input_array in inputs:
            if isinstance(input_array, Timeseries):
                if input_array.units != self.units:
                    raise ValueError('Cannot combine `siffmath.Timeseries` with different units')
        true_args = [
            arg if not isinstance(arg, Timeseries) else arg.view(np.ndarray)
            for arg in inputs
        ]
        return super().__array_ufunc__(ufunc, method, *true_args, **kwargs)

    def shift(self, shift_amount : TimestampLike, units : TimeUnitsLike) -> None:
        """
        Shifts the elements of this timeseries by the amount specified
        in shift_amount, which is in the units specified by `units`.
        """

        units = TimeseriesUnits(units)
        shift_amount = convert_from_unit_to_unit(
            from_units = units,
            to_units = self.units,
            value = shift_amount,
        )

        if shift_amount > 0:
            self.__array__()[:] += shift_amount
        elif shift_amount < 0:
            self.__array__()[:] -= -shift_amount

    def convert_units(self, new_units : TimeUnitsLike) -> None:
        """
        Convert the units of the timeseries
        """
        new_units = TimeseriesUnits(new_units)
        if new_units == self.units:
            return None
        else:
            if new_units == TimeseriesUnits.experiment_seconds:
                raise ValueError(
                    "Cannot convert to experiment_seconds because we don't know the time zero of the experiment. "
                    + "If you know the time zero, use `convert_from_unit_to_unit` instead "
                    + "with the appropriate time zero for the input and output units."
                )
            elif new_units == TimeseriesUnits.epoch_nanoseconds:
                self.__array__()[:] *= 1e9
                self.units = new_units
            elif new_units == TimeseriesUnits.epoch_seconds:
                self.__array__()[:] /= 1e9
                self.units = new_units
            else:
                raise ValueError(f'Unknown units {new_units}')