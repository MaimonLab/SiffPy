"""
Got tired of all the various timeseries types I'm using and
their confusing units. This is a simply numpy array extension
that simply adds units, a few extra fields, and provides
methods to interconvert by units
"""
from typing import Union
from enum import Enum

import numpy as np

TimestampLike = Union[int, float, 'np.dtype[np.int_]', 'np.dtype[np.float_]']

class TimeseriesUnits(Enum):
    """
    Enum for timeseries units
    """
    experiment_seconds = 'experiment_seconds'
    epoch_nanoseconds = 'epoch_nanoseconds'
    epoch_seconds = 'epoch_seconds'

# A number that can be cast to a TimeseriesUnits
TimeUnitsLike = Union[TimeseriesUnits, str]

def convert_from_unit_to_unit(
    from_units : TimeUnitsLike,
    to_units : TimeUnitsLike,
    value : np.ndarray,
    time_zero_from : TimestampLike,
    time_zero_to : TimestampLike,
) -> TimestampLike:
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

    time_zero_from : TimestampLike
        The time zero of the input array
    """
    from_units = TimeseriesUnits(from_units)
    to_units = TimeseriesUnits(to_units)

    if from_units == to_units:
        return value
    
    raise NotImplementedError('Not implemented yet')
    
    #if from_units == TimeseriesUnits.experiment_seconds:
    

class Timeseries(np.ndarray):
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
        units : TimeUnitsLike = TimeseriesUnits.experiment_seconds,
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
    
    # def convert_units(self, new_units : TimeUnitsLike):
    #     """
    #     Convert the units of the timeseries
    #     """
    #     new_units = TimeseriesUnits(new_units)
    #     if new_units == self.units:
    #         return self
    #     else:
    #         if new_units == TimeseriesUnits.experiment_seconds:
    #             return Timeseries(
    #                 self - self.time_zero,
    #                 units = new_units,
    #                 time_zero = 0,
    #             )
    #         elif new_units == TimeseriesUnits.epoch_nanoseconds:
    #             return Timeseries(
    #                 (self - self.time_zero) * 1e9,
    #                 units = new_units,
    #                 time_zero = 0,
    #             )
    #         elif new_units == TimeseriesUnits.epoch_seconds:
    #             return Timeseries(
    #                 (self - self.time_zero) * 1e-9,
    #                 units = new_units,
    #                 time_zero = 0,
    #             )
    #         else:
    #             raise ValueError(f'Unknown units {new_units}')