import numpy as np
from typing import TYPE_CHECKING, Optional, Union
from pathlib import Path
import copy
from enum import Enum

import h5py

from siffpy.siffmath.fluorescence.traces import FluorescenceTrace
from siffpy.core.flim import FlimUnits, convert_flimunits
from siffpy.core.flim import FLIMParams
    

if TYPE_CHECKING:
    from siffpy.core.flim.flimunits import FlimUnitsLike
    from siffpy.core.utils.types import PathLike
    from siffpy.siffmath.utils.types import (
        FlimArrayLike, FluorescenceArrayLike
    )

class FlimMethod(Enum):
    """
    Options for the method attribute of a FlimTrace.

    Determines which operations can be performed and,
    in some cases, what the nature of that transformation
    is. Will grow as more methods are implemented.
    """
    EMPIRICAL = "empirical lifetime"
    PHASOR = "phasor"


class FlimTrace(np.ndarray):
    """
    Subclasses numpy arrays and adds
    extra attributes that might be useful
    for FLIM timeseries and tracking the relationship
    between the lifetime and intensity measures. Useful
    to avoid sacrificing the benefits of numpy while still
    treating FLIM data appropriately. Raw lifetime data can
    be accessed with the property "lifetime" to avoid all of
    the actual FLIM-like math assurances. Fluorescence data
    can be returned as a FluorescenceTrace with the property
    fluorescence.

    Most numpy functions will actually operate on the INTENSITY
    array, altering the FlimTrace core array only if those operations
    make sense.

    The intensity data _will_ be cast to an array of FLOATS to make
    computation much faster. If you do not want to cast to floats,
    use the nocast kwarg in the constructor.
    
    Modeled after
    the RealisticInfoArray example provided at
    https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    
    """

    FILETAG = 'flim_hdf5'

    def __new__(
            cls,
            input_array : 'FlimArrayLike', # INPUT_ARRAY IS THE LIFETIME METRIC
            intensity : Optional['FluorescenceArrayLike'] = None,
            confidence : Optional[np.ndarray] = None,
            FLIMParams : Optional['FLIMParams'] = None,
            method : Optional[str] = None,
            angle : Optional[float] = None,
            units : 'FlimUnitsLike' = FlimUnits.UNKNOWN,
            nocast : bool = False,
            info_string : str = "", # new attributes TBD?
        ):
        """ 
        nocast : bool
            If True, the intensity array will not be cast to float.
            Makes the operations run slower, but preserves the original
            data type.

        WARNING: CONFIDENCE NOT IMPLEMENTED, MUST BE NONE (FOR NOW)
        """
        
        if hasattr(input_array, '__iter__') and all(isinstance(x, FlimTrace) for x in input_array):
            intensity = np.asarray([x.intensity for x in input_array])
            confidence = None
            if all(x.FLIMParams == input_array[0].FLIMParams for x in input_array):
                FLIMParams = input_array[0].FLIMParams
            if all(x.method == input_array[0].method for x in input_array):
                method = input_array[0].method
            if all(x.units == input_array[0].units for x in input_array):
                units = input_array[0].units
            info_string = "[" + " , ".join(x.info_string for x in input_array) + "]"
            input_array = np.asarray([x.__array__() for x in input_array])
            
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asanyarray(input_array).view(cls)
        if intensity is None:
            obj.intensity = np.zeros_like(input_array)
        else:
            obj.intensity = np.asarray(intensity)

        if not nocast:
            obj.intensity = obj.intensity.astype(float)
        
        obj.confidence = confidence
        if obj.confidence is not None:
            raise ValueError("Confidence parameter in FlimTrace not yet implemented.")

        if not (obj.intensity.shape == obj.__array__().shape):
            raise ValueError("Provided intensity and lifetime arrays must have same shape!")

        # add the new attributes to the created instance
        obj.FLIMParams = FLIMParams
        obj.method = FlimMethod(method) if method is not None else None
        obj.angle = angle
        if units is None:
            units = FlimUnits.UNKNOWN
        obj.units = FlimUnits(units)
        obj.info_string = info_string
        
        # Finally, we must return the newly created object:
        return obj

    @property
    def lifetime(self)->np.ndarray:
        """
        Returns a pointer to the FLIM data of a
        FlimArray as a regular numpy array. I think
        this might be dangerous but... leaving it as
        a copy means all the 'intuitive' code one
        might write becomes very slow because it makes
        copy after copy after copy...
        """
        return self[...]

    @property
    def fluorescence(self)->FluorescenceTrace:
        """ Returns the intensity array of a FlimTrace as a FluorescenceTrace """
        return FluorescenceTrace(self.intensity, method = 'Photon counts', F = self.intensity)
    
    def subtract_noise(
            self,
            max_arrival_time : float,
            units : Optional['FlimUnitsLike'] = None,
            n_photons : Union[Optional[int], Optional[np.ndarray]] = None,
        ):
        """
        Either subtracts the noise from the `FLIMParams` fit or
        subtracts an external value of photons from the intensity
        and lifetime data (determined by the `n_photons` parameter).

        Mode 1: `n_photons` is `None`:
        ------------------------------

        If the current `FLIMParams` attribute has a `noise` parameter that may be
        influencing the lifetime value, this subtracts out the estimated noise from
        each array entry by presuming the `noise` value corresponds to that fraction
        of photons originated from a source of uniformly distributed arrival times.

        If self.`FLIMParams` is `None` and `n_photons` is `None, this
        function does nothing. Otherwise, it
        modifies the `intensity` and `lifetime` attributes in place then
        adjusts the `noise` value of the current `FLIMParams` attribute to 0.
        Because this _mutates_ the `FLIMParams` object, this makes a copy so that
        it does not affect other arrays pointing to the same object.

        The subtraction does the following:

        - `intensity` : Multiplies by (1-`noise`) if `n_photons` is `None`.

        - `lifetime` : Subtracts `max_arrival_time/2 * noise`

        Mode 2: `n_photons` is not `None`:
        ----------------------------------

        Subtracts a fixed number of photons from the intensity and lifetime data
        (this can either be an array of the same shape as the current array, or
        a single integer for all time points). This is useful for subtracting
        a time-varying background signal (e.g. projector noise that turns on
        mid-experiment) from the data.

        ## Arguments

        - `max_arrival_time` : float

            In the same units as the current lifetime if `units` is `None`
        
        - `units` : `FlimUnitsLike | None`

            Specifies the units of `max_arrival_time`. If `None`, they are
            presumed to be the same as the `FlimTrace`.
        """
        if n_photons is not None:
            if (
                isinstance(n_photons, np.ndarray) 
                and n_photons.shape != self.intensity.shape
            ):
                raise ValueError("n_photons must be either an integer \
                                 or an array of the same shape as the intensity array."
                )

            if self.method == FlimMethod.EMPIRICAL:
                self[...] -= max_arrival_time*n_photons/(2*(n_photons + self.intensity))
            elif self.method == FlimMethod.PHASOR:
                raise NotImplementedError("Subtracting noise from phasor methods is not yet implemented.")
            else:
                raise NotImplementedError("Subtracting noise from methods other than `empirical lifetime`"
                            +  "is not yet implemented.")
            self.intensity -= n_photons
            return

        if self.FLIMParams is None:
            return
        if self.FLIMParams.noise == 0:
            return
        if units is not None:
            if self.units is None:
                raise ValueError(
                    "If `max_arrival_time` units are specified in `subtract_noise`," \
                    + " the FlimTrace itself must have units to convert it into."
                )
            max_arrival_time = convert_flimunits(
                max_arrival_time,
                from_units = FlimUnits(units),
                to_units = self.units
            )
        max_arrival_time = float(max_arrival_time)
        self[...] -= (self.FLIMParams.noise)*max_arrival_time/2
        self[...] /= 1-self.FLIMParams.noise
        self.intensity *= (1-self.FLIMParams.noise)
        self.FLIMParams : FLIMParams = copy.deepcopy(self.FLIMParams)
        self.FLIMParams.noise = 0.0

    def set_units(self, units : 'FlimUnitsLike'):
        """
        Allows setting of units even if the units are unknown. Basically
        a forced version of `convert_units`
        """
        if self.units == FlimUnits.UNKNOWN:
            self.units = FlimUnits(units)
        else:
            self.convert_units(units)

    def convert_units(self, units : 'FlimUnitsLike'):
        """ Converts units in place """
        
        self[...] = convert_flimunits(self.__array__(), self.units, units)
        self.units = units
        if self.FLIMParams is not None:
            self.FLIMParams.convert_units(units)
    
    @property
    def _inheritance_dict(self)->dict:
        return {
            'confidence' : self.confidence,
            'FLIMParams' : self.FLIMParams,
            'method' : self.method,
            'angle' : self.angle,
            'info_string' : self.info_string,
            'units' : self.units,
        }

    def __repr__(self)->str:
        return f"{self.__class__.__name__} :\n" + \
        f"Units : {self.units}, Info: {self.info_string}, Method : {self.method}\n"+\
        f"Lifetime:\n{self.__array__()}\nIntensity:\n{self.intensity}"

    def __array_wrap__(self, out_arr, context=None):
        return super().__array_wrap__(out_arr, context=context)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.intensity = getattr(obj, 'intensity', np.full_like(self.__array__(), np.nan))
        self.confidence = getattr(obj, 'confidence', None)
        self.FLIMParams = getattr(obj, 'FLIMParams', None)
        self.method = getattr(obj, 'method', None)
        self.angle = getattr(obj, 'angle', None)
        self.units = getattr(obj, 'units', FlimUnits.UNKNOWN)
        self.info_string = getattr(obj,'info_string', '')

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        When you use ufuncs on FlimTraces,
        the array itself and its intensity / confidence
        measures should behave accordingly. The method, FLIMParams,
        angle, and info_string are all inherited so long as they're
        consistent across all FlimTraces involved
        """
        
        if (ufunc in FlimTrace.EXCEPTED_UFUNCS) and (method in FlimTrace.EXCEPTED_UFUNCS[ufunc]):
            return NotImplemented

        inp_args = [] # convert to a regular array
        for input_ in inputs:
            if isinstance(input_, FlimTrace):
                inp_args.append(input_.view(np.ndarray))
            else:
                inp_args.append(input_)

        # if it's not handled, just treat it like a regular array, operating on the FLIM data. DANGEROUS but seems simplest
        if not (
            (ufunc in FlimTrace.HANDLED_UFUNCS) and (method in FlimTrace.HANDLED_UFUNCS[ufunc])
        ):
            return super().__array_ufunc__(ufunc, method, *inp_args, out = out, **kwargs)

        return FlimTrace.HANDLED_UFUNCS[ufunc][method](*inputs, out = out, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """
        Some functions get special implementations, others just
        have this be treated like a numpy array, and others still
        are explicitly excepted.
        """
        if func not in FlimTrace.HANDLED_FUNCTIONS:
            if func in FlimTrace.EXCEPTED_FUNCTIONS:
                return NotImplemented
            replaced_args = [arg.__array__() if isinstance(arg, self.__class__) else arg for arg in args]
            return func._implementation(*replaced_args, **kwargs)
        return FlimTrace.HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __getitem__(self, key)->'FlimTrace':
        """
        Returns the indexed version of BOTH arrays
        """
        if key is ...:
            return self.__array__()[...]
        else:
            return FlimTrace(
                self.__array__()[key],
                self.intensity[key],
                **self._inheritance_dict
            )            

    def _equal_params(self, other)->bool:
        """ Checks that two FlimTraces have compatible params """
        if isinstance(other, FlimTrace):
            if (
                (self.FLIMParams == other.FLIMParams) and
                (self.method == other.method)
            ):
                return True
        return False

    # NUMPY METHODS IMPLEMENTED BY ARRAYS
    # ANNOYING TO HAVE TO OVERWRITE THEM BUT IT MAKES SENSE WITH PYTHON
    def sum(self, *args, **kwargs):
        return np.sum(self, *args, **kwargs)

    def nansum(self, *args, **kwargs):
        return np.nansum(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        return np.mean(self, *args, **kwargs)

    def nanmean(self, *args, **kwargs):
        return np.nanmean(self, *args, **kwargs)

    def reshape(self, *shape, **kwargs):

        # A little special, since the array class implementation
        # of reshape is willing to accept several arguments and
        # treat them like a tuple.
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return np.reshape(self, shape, **kwargs)

    def append(self, *args, **kwargs):
        return np.append(self, *args, **kwargs)
    
    def ravel(self, *args, **kwargs):
        return np.ravel(self, *args, **kwargs)

    def sort(self, *args, sortby = 'flim', **kwargs):
        """ Sortby argument must be either 'flim' or 'intensity' """
        return np.sort(self, *args, sortby = sortby, **kwargs)

    def squeeze(self, *args, **kwargs):
        return np.squeeze(self, *args, **kwargs)

    def transpose(self, *args, **kwargs):
        return np.transpose(self, *args, **kwargs)

    # populated in siffpy.siffmath.flim.trace_funcs.flimtrace_ufuncs
    HANDLED_UFUNCS = {

    }

    EXCEPTED_UFUNCS = {

    }

    # populated in siffpy.siffmath.flim.trace_funcs.flimtrace_funcs
    HANDLED_FUNCTIONS = {

    }

    EXCEPTED_FUNCTIONS = [

    ]

    @classmethod
    def implements_func(cls, *np_functions):
        """Register an __array_function__ implementation for FlimTrace objects."""
        def decorator(func):
            for np_function in np_functions:
                FlimTrace.HANDLED_FUNCTIONS[np_function] = func
            return func
        return decorator

    @classmethod
    def excludes_func(cls, *np_functions):
        """ Excludes a function from being used on a FlimTrace object """
        for np_function in np_functions:
            FlimTrace.EXCEPTED_FUNCTIONS.append(np_function)

    @classmethod
    def implements_ufunc(cls, np_ufunction, method):
        """Register an __array_ufunction__ implementation for FlimTrace objects."""
        def decorator(func):
            if np_ufunction not in FlimTrace.HANDLED_UFUNCS:
                FlimTrace.HANDLED_UFUNCS[np_ufunction] = {}
            FlimTrace.HANDLED_UFUNCS[np_ufunction][method] = func
            return func
        return decorator

    @classmethod
    def excludes_ufunc(cls, np_ufunction, method):
        """ Excludes a ufunction from being used on a FlimTrace object """
        if np_ufunction not in FlimTrace.EXCEPTED_UFUNCS:
            FlimTrace.EXCEPTED_UFUNCS[np_ufunction] = []
        FlimTrace.EXCEPTED_UFUNCS[np_ufunction].append(method)

    def save(self, path : 'PathLike'):
        path = Path(path)
        path = path.with_suffix(f'.{self.__class__.FILETAG}')
        with h5py.File(path, 'w') as f:
            f.create_dataset("flim", data = self.__array__())
            f.create_dataset("intensity", data = self.intensity)
            #f['FLIMParams'] = self.FLIMParams can't store arbitrary Python object...
            f.attrs['units'] = h5py.Empty('s') if self.units is None else FlimUnits(self.units).value
            f.attrs['method'] = h5py.Empty('s') if self.method is None else FlimMethod(self.method).value
            f.attrs['angle'] = h5py.Empty('f') if self.angle is None else self.angle
            f.attrs['info_string'] = h5py.Empty('s') if self.info_string is None else self.info_string
        if self.FLIMParams is not None:
            self.FLIMParams.save(path.with_suffix('.flim_params'))

    @classmethod
    def load(cls, path : 'PathLike')->'FlimTrace': 
        """ Load a .flim_hdf file and create a FlimArray class from it. """
        path = Path(path)
        if not path.suffix == f'.{cls.FILETAG}':
            raise ValueError(f"File must be a .{cls.FILETAG} file")

        with h5py.File(path, 'r') as f:
            flim = f['flim'][...]
            intensity = f['intensity'][...]
            _FLIMParams = FLIMParams.load(path.with_suffix('.flim_params'))
            method = None if isinstance(f.attrs['method'], h5py.Empty) else f.attrs['method']
            angle = None if isinstance(f.attrs['angle'], h5py.Empty) else f.attrs['angle']
            info_string = None if isinstance(f.attrs['info_string'], h5py.Empty) else f.attrs['info_string']
            units = (
                None
                if ('units' not in f.attrs) or isinstance(f.attrs['units'], h5py.Empty)
                else f.attrs['units']
            )

        return cls(
            flim,
            intensity = intensity,
            FLIMParams = _FLIMParams,
            method = method,
            angle = angle,
            units = _FLIMParams.units if units is None else FlimUnits(units),
            info_string = info_string
        )
    
    def to_alpha(self, threshold : int, scale : float)->np.ndarray:
        """
        Returns an array that can be used an as alpha channel for
        imshow (i.e. 0 to 1 scaled by intensity of the array). Operation
        is (intensity - threshold) * scale, clipped to 0 and 1.
        """
        return np.clip(scale*(self.intensity - threshold), 0, 1)
