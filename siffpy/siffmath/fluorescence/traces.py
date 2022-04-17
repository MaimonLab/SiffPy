import numpy as np
from typing import Union

from ..utils import fifth_percentile

class FluorescenceTrace(np.ndarray):
    """
    Extends the numpy array to provide
    extra attributes that might be useful
    for parsing the trace data. Behaves like
    a normal numpy array. Modeled after
    the RealisticInfoArray example provided at
    https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    """

    VECTOR_PROPERTIES = [
        'F', 'F0', 'max_val', 'min_val'
    ]

    LIST_PROPERTIES = [
        'method', 'normalized', 'info_string', 'angle'
    ]

    def __new__(cls, input_array, method : str = None, normalized : bool = False,
        F : np.ndarray = None, F0 : np.ndarray = np.ndarray(None),
        max_val : np.ndarray = np.inf , min_val : np.ndarray = 0.0,
        angle : float = None,
        info_string : str = None, # new attributes TBD?
        ):
        
        if isinstance(input_array, (list,tuple)):
            if all(isinstance(x, FluorescenceTrace) for x in input_array):
                return FluorescenceVector(input_array)
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        
        # add the new attributes to the created instance
        obj.method = method
        obj.normalized = normalized
        if F is None:
            F = np.zeros_like(input_array)
        obj.F = F
        obj.F0 = F0
        obj.max_val = max_val
        obj.min_val = min_val
        obj.angle = angle
        obj.info_string = info_string
        

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.method = getattr(obj, 'method', None)
        self.normalized = getattr(obj, 'normalized', False)
        self.F = getattr(obj,'F', np.zeros_like(obj.__array__))
        self.F0 = getattr(obj, 'F0', 0)
        self.max_val = getattr(obj, 'max_val', np.inf)
        self.min_val = getattr(obj, 'min_val', 0)
        self.angle = getattr(obj, 'angle', None)
        self.info_string = getattr(obj,'info_string', '')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        When you use ufuncs on FluorescenceTraces,
        their vectorlike properties should behave accordingly
        """
        
        np_args = [] # args as numpy arrays for the ufunc call
        ftraces = [] # args that are FluorescenceTraces
        for input_ in inputs:
            if isinstance(input_, FluorescenceTrace):
                np_args.append(input_.view(np.ndarray))
                ftraces.append(input_)
            else:
                np_args.append(input_)

        results = super().__array_ufunc__(ufunc, method, *np_args, **kwargs)

        # Iterate over vector properties and perform the ufunc on each
        prop_results = {}
        for prop in FluorescenceTrace.VECTOR_PROPERTIES:
            # construct the prop arguments for the ufunc
            prop_args = []
            for input_ in inputs:
                if isinstance(input_, FluorescenceTrace):
                    prop_args.append(getattr(input_, prop))
                else:
                    prop_args.append(input_)
            try:
                prop_results[prop] = super().__array_ufunc__(ufunc, method, *prop_args, **kwargs)
            except:
                continue
        
        # list results can be shared if they're consonant across all args
        list_results = {}
        for prop in FluorescenceTrace.LIST_PROPERTIES:
            if all(getattr(trace, prop) == getattr(ftraces[0], prop) for trace in ftraces):
                list_results[prop] = getattr(ftraces[0],prop)

        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = np.asarray(results).view(FluorescenceTrace)
            
            for key, val in prop_results.items():
                setattr(results, key, val)

            for key,val in list_results.items():
                setattr(results, key, val)

            results = (results,)
        
        else:
            resultlist = []
            for idx, res in enumerate(results):    
                resultlist.append(np.asarray(res).view(FluorescenceTrace))
                for prop,val in prop_results.items():
                    setattr(resultlist[idx], prop, val[idx])
                # TODO: what should I do with list attributes in this case?
                # TODO: should they just propagate?
                for prop, val in list_results.items():
                    setattr(resultlist[idx], prop, val[idx])

            results = tuple(resultlist)

        return results[0] if len(results) == 1 else results


class FluorescenceVector(FluorescenceTrace):
    """
    Constructed if a FluorescenceTrace is made out of an iterable of FluorescenceTraces.

    A special type of FluorescenceTrace that keeps track of the fact that each of its major
    dimension is supposed to be its own FluorescenceTrace object.
    """

    def __new__(cls, input_array : Union[list[FluorescenceTrace],tuple[FluorescenceTrace]]):
        
        # converts the input iterable into a standard vector one dimension larger
        obj = np.asarray(input_array).view(cls)

        # Make the vector inherit properties in a reasonable way

        def listify_attribute(attr: str):
            return [getattr(x, attr) for x in input_array]

        def numpify_attribute(attr: str):
            return np.array(listify_attribute(attr))
        
        for prop in FluorescenceTrace.VECTOR_PROPERTIES:
            setattr(obj, prop, numpify_attribute(prop))

        for prop in FluorescenceTrace.LIST_PROPERTIES:
            setattr(obj, prop, listify_attribute(prop)) 

        # Finally, we must return the newly created object:
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        When you use ufuncs on FluorescenceVectors,
        their vectorlike properties should behave accordingly
        """
        
        np_args = [] # args as numpy arrays for the ufunc call
        ftraces = [] # args that are FluorescenceTraces
        for input_ in inputs:
            if isinstance(input_, FluorescenceVector):
                np_args.append(input_.view(FluorescenceTrace))
                ftraces.append(input_)
            else:
                np_args.append(input_)

        results = FluorescenceTrace.__array_ufunc__(self, ufunc, method, *np_args, **kwargs)

        # Iterate over vector properties and perform the ufunc on each
        prop_results = {}
        for prop in FluorescenceTrace.VECTOR_PROPERTIES:
            # construct the prop arguments for the ufunc
            prop_args = []
            for input_ in inputs:
                if isinstance(input_, FluorescenceTrace):
                    prop_args.append(getattr(input_, prop))
                else:
                    prop_args.append(input_)
            try:
                prop_results[prop] = getattr(ufunc,method)(*prop_args, **kwargs) # uses numpy array's ufunc, not FluorescenceTrace's
            except:
                continue
        
        # list results can be shared if they're consonant across all args
        list_results = {}
        for prop in FluorescenceTrace.LIST_PROPERTIES:
            if all(getattr(trace, prop) == getattr(ftraces[0], prop) for trace in ftraces):
                list_results[prop] = getattr(ftraces[0],prop)

        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = np.asarray(results).view(FluorescenceVector)
            for key, val in prop_results.items():
                setattr(results, key, val)

            for key,val in list_results.items():
                setattr(results, key, val)

            results = (results,)
        
        else:
            resultlist = []
            for idx, res in enumerate(results):    
                resultlist.append(np.asarray(res).view(FluorescenceVector))
                for prop,val in prop_results.items():
                    setattr(resultlist[idx], prop, val[idx])
            results = tuple(resultlist)

        return results[0] if len(results) == 1 else results

    def __getitem__(self, key):
        """
        Returns instances of FluorescenceTrace for int keys.
        """
        if isinstance(key, int):
            rettrace = FluorescenceTrace(self.__array__()[key])
            for prop in FluorescenceTrace.VECTOR_PROPERTIES:
                setattr(rettrace, prop, getattr(self,prop)[key])
        
            for prop in FluorescenceTrace.LIST_PROPERTIES:
                setattr(rettrace, prop, getattr(self, prop)[key])

            return rettrace
        else:
            # retval = super().__getitem__(key)
            # if len(retval.shape) == 1:
            #     rettrace = FluorescenceTrace(retval.__array__())
            #     for prop in FluorescenceTrace.VECTOR_PROPERTIES:
            #         setattr(rettrace, prop, getattr(self,prop)[key])
            #     return rettrace
                
            return np.asarray(super().__getitem__(key))