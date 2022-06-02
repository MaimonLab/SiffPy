from typing import Union
import logging

import numpy as np

from ...siffutils.flimparams import FLIMParams
from ..fluorescence.traces import FluorescenceTrace

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
    
    Modeled after
    the RealisticInfoArray example provided at
    https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    
    FlimTrace(
        input_array,
        intensity : np.ndarray = None,
        confidence : np.ndarray = None,
        FLIMParams : FLIMParams = None,
        method : str = None,
        angle : float = None,
        info_string : str = None
    )
    """

    def __new__(cls, input_array : np.ndarray, # INPUT_ARRAY IS THE LIFETIME METRIC
            intensity : np.ndarray = None,
            confidence : np.ndarray = None,
            FLIMParams : FLIMParams = None,
            method : str = None,
            angle : float = None,
            info_string : str = "", # new attributes TBD?
        ):
        
        if isinstance(input_array, (list,tuple)):
            if all(isinstance(x, FlimTrace) for x in input_array):
                logging.warning("Haven't implemented the FlimVector yet. Returning just a FlimTrace")
                #return FlimVector(input_array)
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        obj.intensity = intensity

        if obj.intensity is None:
            logging.warning("Creating a FlimTrace without an intensity value! Filling with NaNs.")
            obj.intensity = np.full_like(input_array, np.nan)
        obj.confidence = confidence
        if not (obj.confidence is None):
            raise ValueError("Confidence parameter in FlimTrace not yet implemented.")

        if not (obj.intensity.shape == obj.__array__().shape):
            raise ValueError("Provided intensity and lifetime arrays must have same shape!")

        # add the new attributes to the created instance
        obj.FLIMParams = FLIMParams
        obj.method = method
        obj.angle = angle
        obj.info_string = info_string
        
        # Finally, we must return the newly created object:
        return obj

    @property
    def lifetime(self)->np.ndarray:
        """ Returns a COPY of the FLIM data of a FlimArray as a regular numpy array """
        return self.__array__().copy()

    @property
    def fluorescence(self)->FluorescenceTrace:
        """ Returns the intensity array of a FlimTrace as a FluorescenceTrace """
        return FluorescenceTrace(self.intensity, method = 'Photon counts', F = self.intensity)

    @property
    def _inheritance_dict(self)->dict:
        return {
            'confidence' : self.confidence,
            'FLIMParams' : self.FLIMParams,
            'method' : self.method,
            'angle' : self.angle,
            'info_string' : self.info_string
        }

    def __repr__(self)->str:
        return f"{self.__class__.__name__} : {self.info_string}\nLifetime:\n{self.__array__()}\nIntensity:\n{self.intensity}"

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.intensity = getattr(obj, 'intensity', np.full_like(self.__array__(), np.nan))
        self.confidence = getattr(obj, 'confidence', None)
        self.FLIMParams = getattr(obj, 'FLIMParams', None)
        self.method = getattr(obj, 'method', None)
        self.angle = getattr(obj, 'angle', None)
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

    def _equal_params(self, other)->bool:
        """ Checks that two FlimTraces have compatible params """
        if isinstance(other, FlimTrace):
            if (
                (self.FLIMParams == other.FLIMParams) and
                (self.method == other.method)
            ):
                return True
        return False

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
    def implements_func(cls, np_function):
        """Register an __array_function__ implementation for FlimTrace objects."""
        def decorator(func):
            FlimTrace.HANDLED_FUNCTIONS[np_function] = func
            return func
        return decorator

    @classmethod
    def excludes_func(cls, np_function):
        """ Excludes a function from being used on a FlimTrace object """
        def decorator(func):
            FlimTrace.EXCEPTED_FUNCTIONS.append(np_function)
            return func
        return decorator

    @classmethod
    def implements_ufunc(cls, np_ufunction, method):
        """Register an __array_ufunction__ implementation for FlimTrace objects."""
        def decorator(func):
            if not np_ufunction in FlimTrace.HANDLED_UFUNCS:
                FlimTrace.HANDLED_UFUNCS[np_ufunction] = {}
            FlimTrace.HANDLED_UFUNCS[np_ufunction][method] = func
            return func
        return decorator

    @classmethod
    def excludes_ufunc(cls, np_ufunction, method):
        """ Excludes a ufunction from being used on a FlimTrace object """
        def decorator(func):
            if not np_ufunction in FlimTrace.EXCEPTED_UFUNCS:
                FlimTrace.EXCEPTED_UFUNCS[np_ufunction] = []
            FlimTrace.EXCEPTED_UFUNCS[np_ufunction].append(method)
            return func
        return decorator







# class FlimVector(FluorescenceTrace):
# #class FluorescenceVector(np.ndarray):
#     """
#     Constructed if a FluorescenceTrace is made out of an iterable of FluorescenceTraces.

#     A special type of FluorescenceTrace that keeps track of the fact that each of its major
#     dimension is supposed to be its own FluorescenceTrace object.
#     """

#     def __new__(cls, input_array : Union[list[FluorescenceTrace],tuple[FluorescenceTrace]]):
        
#         # converts the input iterable into a standard vector one dimension larger
#         obj = np.asarray(input_array).view(cls)
        
#         # Stops numpy from making FluorescenceVectors where they don't belong
#         if (not obj.shape) or (not all(isinstance(x, FluorescenceTrace) for x in input_array)):
#             return obj.view(np.ndarray)

#         # Make the vector inherit properties in a reasonable way
#         # now that we know they're all FluorescenceTraces

#         def listify_attribute(attr: str):
#             return [getattr(x, attr) for x in input_array]

#         def numpify_attribute(attr: str):
#             return np.array(listify_attribute(attr))
        
#         for prop in FluorescenceTrace.VECTOR_PROPERTIES:
#             setattr(obj, prop, numpify_attribute(prop))

#         for prop in FluorescenceTrace.LIST_PROPERTIES:
#             setattr(obj, prop, listify_attribute(prop)) 

#         # Finally, we must return the newly created object:
#         return obj

#     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
#         """
#         When you use ufuncs on FluorescenceVectors,
#         their vectorlike properties should behave accordingly
#         """
#         np_args = [] # args as numpy arrays for the ufunc call
#         ftraces = [] # args that are FluorescenceTraces
#         for input_ in inputs:
#             if isinstance(input_, (FluorescenceTrace,FluorescenceVector)):
#                 np_args.append(input_.view(FluorescenceTrace))
#                 ftraces.append(input_)
#             else:
#                 np_args.append(input_)

#         results = super().__array_ufunc__(ufunc, method, *np_args, **kwargs)
#         #results = getattr(ufunc, method)(*np_args, **kwargs)

#         # Iterate over vector properties and perform the ufunc on each
#         prop_results = {}
#         for prop in FluorescenceTrace.VECTOR_PROPERTIES:
#             # construct the prop arguments for the ufunc
#             prop_args = []
#             for input_ in inputs:
#                 if isinstance(input_, FluorescenceTrace):
#                     prop_args.append(np.asarray(getattr(input_, prop)))
#                 else:
#                     prop_args.append(input_)
#             try:
#                 prop_results[prop] = getattr(ufunc,method)(*prop_args, **kwargs) # uses numpy array's ufunc, not FluorescenceTrace's
#             except:
#                 continue
        
#         # list results can be shared if they're consonant across all args
#         list_results = {}
#         for prop in FluorescenceTrace.LIST_PROPERTIES:
#             if all(getattr(trace, prop) == getattr(ftraces[0], prop) for trace in ftraces):
#                 list_results[prop] = getattr(ftraces[0],prop)

#         if results is NotImplemented:
#             return NotImplemented

#         if ufunc.nout == 1:
#             results = np.asarray(results).view(FluorescenceVector)
#             for key, val in prop_results.items():
#                 if not (val is NotImplemented):
#                     setattr(results, key, np.asarray(val))

#             for key,val in list_results.items():
#                 setattr(results, key, val)

#             results = (results,)
        
#         else:
#             resultlist = []
#             for idx, res in enumerate(results):    
#                 resultlist.append(np.asarray(res).view(FluorescenceVector))
#                 for prop,val in prop_results.items():
#                     if not (val[idx] is NotImplemented):
#                         setattr(resultlist[idx], prop, np.asarray(val[idx]))
            
#             results = tuple(resultlist)

#         return results[0] if len(results) == 1 else results

#     def __array_finalize__(self, obj):
#         # see InfoArray.__array_finalize__ for comments
#         if obj is None: return
#         curr_array = self.__array__()
#         try:
#             n_trace = curr_array.shape[0]
#         except IndexError:
#             n_trace = 0
#         self.method = getattr(obj, 'method', n_trace*[None])
#         self.normalized = getattr(obj, 'normalized', n_trace*[False])
#         self.F = getattr(obj,'F',  np.full_like(curr_array, np.nan))
#         self.F0 = getattr(obj, 'F0', np.full((n_trace,), np.nan))
#         self.max_val = getattr(obj, 'max_val', np.full((n_trace,), np.inf))
#         self.min_val = getattr(obj, 'min_val', np.zeros((n_trace,)))
#         self.angle = getattr(obj, 'angle', n_trace*[None])
#         self.info_string = getattr(obj,'info_string', n_trace*[''])

#     def __getitem__(self, key):
#         """
#         Returns instances of FluorescenceTrace for int keys.
#         """
#         if isinstance(key, int):
#             rettrace = FluorescenceTrace(self.__array__()[key])
#             for prop in FluorescenceTrace.VECTOR_PROPERTIES:
#                 setattr(rettrace, prop, getattr(self,prop)[key])
        
#             for prop in FluorescenceTrace.LIST_PROPERTIES:
#                 setattr(rettrace, prop, getattr(self, prop)[key])

#             return rettrace
#         else:
#             # retval = super().__getitem__(key)
#             # if len(retval.shape) == 1:
#             #     rettrace = FluorescenceTrace(retval.__array__())
#             #     for prop in FluorescenceTrace.VECTOR_PROPERTIES:
#             #         setattr(rettrace, prop, getattr(self,prop)[key])
#             #     return rettrace
                
#             return np.asarray(super().__getitem__(key))