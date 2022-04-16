"""
Dedicated code for data that is purely fluorescence analysis
"""
import numpy as np
import inspect

from .utils import fifth_percentile

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
        'method', 'normalized', 'info_string'
    ]

    def __new__(cls, input_array, method : str = None, normalized : bool = False,
        F : np.ndarray = None, F0 : np.ndarray = np.ndarray(None),
        max_val : np.ndarray = np.inf , min_val : np.ndarray = 0.0,
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
            results = results.view(FluorescenceTrace)
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
            results = tuple(resultlist)

        return results[0] if len(results) == 1 else results


class FluorescenceVector(FluorescenceTrace):
    """
    Constructed if a FluorescenceTrace is made out of an iterable of FluorescenceTraces
    """

    def __new__(cls, input_array : list[FluorescenceTrace]):
        
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

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.method = getattr(obj, 'method', None)
        self.normalized = getattr(obj, 'normalized', False)
        self.F = getattr(obj,'F', 0)
        self.F0 = getattr(obj, 'F0', 0)
        self.max_val = getattr(obj, 'max_val', 0)
        self.min_val = getattr(obj, 'min_val', 0)
        self.info_string = getattr(obj,'info_string', '')

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
            results = results.view(FluorescenceVector)
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

        

def photon_counts(fluorescence : np.ndarray, *args, **kwargs)->FluorescenceTrace:
    """ Simply returns raw photon counts. This is just the array that's passed in, wrapped in a FluorescenceTrace. """
    return FluorescenceTrace(fluorescence, F = fluorescence, method = 'Photon counts', F0 = 0)

def dFoF(fluorescence : np.ndarray, *args, normalized : bool = False, Fo = fifth_percentile, **kwargs)->FluorescenceTrace:
    """
    
    Takes a numpy array and returns a dF/F0 trace across the rows -- i.e. each row is normalized independently
    of the others. Returns a version of the function (F - F0)/F0, where F0 is computed as below

    fluorescence : np.ndarray

        The data constituting the F in dF/F0

    normalized : bool (optional)

        Compresses the response of each row to approximately the range 0 - 1 (uses the 5th and 95th percentiles).
        Default is False

    Fo : callable or np.ndarray (optional)

        How to determine the F0 term for a given row. If Fo is callable, the function is applied to the
        roi numpy array directly (i.e. it's NOT a function that operates on only one row at a time). 
        Can also provide just a number or an array of numbers.

    Passes additional args and kwargs to the Fo function, if those args and kwargs are provided.
    
    """
    if not isinstance(fluorescence,np.ndarray):
        fluorescence = np.array(fluorescence)
    fluorescence = np.atleast_2d(fluorescence)
    
    #info_string = ""
    if callable(Fo):
        F0 = Fo(fluorescence, *args, **kwargs)
        #inspect.signature(Fo).
    elif type(Fo) is np.ndarray or float:
        F0 = Fo
    else:
        try:
            np.array(Fo).astype(float)
        except TypeError:
            raise TypeError(f"Keyword argument Fo is not of type float, a numpy array, or a callable, nor can it be cast to such.")

    df_trace = ((fluorescence.T.astype(float) - F0)/F0).T
    max_val = None
    min_val = None
    
    if normalized:
        sorted_vals = np.sort(df_trace,axis=1)
        min_val = sorted_vals[:,sorted_vals.shape[-1]//20]
        max_val = sorted_vals[:,int(sorted_vals.shape[-1]*(1.0-1.0/20))]
        df_trace = ((df_trace.T - min_val)/(max_val - min_val)).T
    
    return FluorescenceTrace(
        df_trace, normalized = normalized, method = 'dF/F', 
        F = fluorescence, F0 = F0,
        max_val = max_val, min_val = min_val
    )

def roi_masked_fluorescence_numpy(frames : np.ndarray, rois : list[np.ndarray])->np.ndarray:
    """
    Takes an array of frames organized as a k-dimensional numpy array with the
    last three dimensions being ('time', 'y', 'x') and converts them into an k-2
    dimensional array, with the final two dimensions of 'frames' compressed against
    the masks in rois. For use on arrays not generated by `SiffPy` and/or not using
    the `siffpy.siffplot.ROI` class
    """
    rois = np.array(rois)
    return np.sum(
            np.tensordot(
                frames,
                rois,
                axes=((-1,-2),(-1,-2))
            ),
            axis = (-2,-1)
        )
