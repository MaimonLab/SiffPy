"""
Array ufunctions for FlimTraces

To be added to whenever I encounter something strange.
"""
import logging

import numpy as np

from ..traces import FlimTrace
from ..flimunits import FlimUnits

### CALL_UFUNC #####

def flimtrace_ufunc_call_pattern(numpy_ufunc : np.ufunc, *args, out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
    """
    A pattern for calling ufuncs on FlimTrace data that do not perform addition-like phenomena
    
    The general procedure:

    Case 1: If only one of *args are a FlimTrace, apply the ufunc to the
    intensity variable of the FlimTrace and the non-FlimTrace argument. Then
    return a FlimTrace that inherits the other properties of the FlimTrace provided
    (or store in 'out'). Behaves like the typical '__call__' pattern with the same arguments.

    Case 2: All are FlimTraces. Return NotImplemented.
    """
    ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}

    if all(isinstance(x,FlimTrace) for x in args):
        logging.warning(
            f"numpy ufunc {numpy_ufunc.__name__} not implemented for multiple FlimTraces. "
            "If you need this operation, view as a np.ndarray with .view() method."
        )
        return NotImplemented

    flimtrace = next((x for x in args if isinstance(x, FlimTrace)))
    new_args = [x.intensity if isinstance(x, FlimTrace) else x for x in args]

    if out is None:    
        return FlimTrace(
            flimtrace.__array__(),
            intensity = numpy_ufunc(*new_args, out=None, **ufunc_kwargs, **kwargs),
            **flimtrace._inheritance_dict
        )
    if not isinstance(out, FlimTrace):
        logging.warning(
            "Calling numpy {np_ufunc.__name__} on a FlimTrace but specifying a return value that is not a FlimTrace. "
            "Performing standard numpy {np_ufunc.__name__} on the intensity data.",
            stacklevel=2
        )
        return numpy_ufunc(*new_args, out = out, **ufunc_kwargs, **kwargs)
    
    return numpy_ufunc(*new_args, out = out.intensity, **ufunc_kwargs)

@FlimTrace.implements_ufunc(np.add, "__call__")
def add_call_flimtrace(x1, x2, /, out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
    """ Adds directly to the intensity if a non-FlimTrace is provided """
    
    ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}
    retval = flimtrace_ufunc_call_pattern(np.add, x1, x2, out=out, **ufunc_kwargs, **kwargs)
    
    if not (retval is NotImplemented):
        return retval

    # Both are FlimTraces, now we can do something actually interesting!
    x1 : FlimTrace
    x2 : FlimTrace

    if not (x1.units == x2.units):
        if any(x.units == FlimUnits.UNKNOWN for x in [x1,x2]): # means they're not BOTH unknown
            raise ValueError("Unable to add FlimTraces with incompatible units")
        try:
            x2.convert_units(x1.units)
        except ValueError:
            x1.convert_units(x2.units)

    new_intensity = np.add(x1.intensity, x2.intensity, out=None, **ufunc_kwargs, **kwargs)
    new_lifetime = np.add(
        np.multiply(np.asarray(x1), x1.intensity, out=None, **ufunc_kwargs, **kwargs),
        np.multiply(np.asarray(x2), x2.intensity, out=None, **ufunc_kwargs, **kwargs),
        out = None,
        **ufunc_kwargs,
        **kwargs
    )

    new_lifetime = np.divide(new_lifetime, new_intensity, out=None, **ufunc_kwargs, **kwargs) # can't seem to do inplace for some reason?

    if out is None:
        params = {}
        if x1._equal_params(x2):
            params = x1._inheritance_dict
        
        return FlimTrace(
            new_lifetime,
            intensity = new_intensity,
            **params
        )
    
    if not isinstance(out, FlimTrace):
        return NotImplemented
    
    out[...] = new_lifetime
    out.intensity = new_intensity
    return out

@FlimTrace.implements_ufunc(np.subtract, "__call__")
def subtract_call_flimtrace(x1, x2, /, out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
    """ Haven't decided if this implementation actually makes sense! """
    ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}
    retval = flimtrace_ufunc_call_pattern(np.subtract, x1, x2, out=out, **ufunc_kwargs, **kwargs)
    
    if not (retval is NotImplemented):
        return retval

    # Both are FlimTraces, now we can do something actually interesting!
    x1 : FlimTrace
    x2 : FlimTrace

    new_intensity = np.subtract(x1.intensity, x2.intensity, out=None, **ufunc_kwargs, **kwargs)
    new_lifetime = np.subtract(
        np.multiply(np.asarray(x1), x1.intensity, out=None, **ufunc_kwargs, **kwargs),
        np.multiply(np.asarray(x2), x2.intensity, out=None, **ufunc_kwargs, **kwargs),
        out = None,
        **ufunc_kwargs,
        **kwargs
    )

    new_lifetime = np.divide(new_lifetime, new_intensity, out=None, **ufunc_kwargs, **kwargs) # can't seem to do inplace for some reason?

    if out is None:
        params = {}
        if x1._equal_params(x2):
            params = x1._inheritance_dict
        
        return FlimTrace(
            new_lifetime,
            intensity = new_intensity,
            **params
        )
    
    if not isinstance(out, FlimTrace):
        return NotImplemented
    
    out[...] = new_lifetime
    out.intensity = new_intensity
    return out

@FlimTrace.implements_ufunc(np.multiply, "__call__")
def multiply_call_flimtrace(x1, x2, /, out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
    """ Cannot multiply two FlimTrace """
    ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}
    return flimtrace_ufunc_call_pattern(np.multiply, x1, x2, out=out, **ufunc_kwargs, **kwargs)

@FlimTrace.implements_ufunc(np.matmul, "__call__")
def matmul_call_flimtrace(x1, x2, /, out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):

    ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}
    retval = flimtrace_ufunc_call_pattern(np.multiply, x1, x2, out=out, **ufunc_kwargs, **kwargs)
    
    if not (retval is NotImplemented):
        return retval

    # Both are FlimTraces! Now we can perform the matrix multiplication, which is an array of sums,
    # appropriately, inheriting the lifetime values that accompany performing such a sum.

    x1 : FlimTrace
    x2 : FlimTrace

    new_intensity = np.matmul(x1.intensity, x2.intensity, out=None, **ufunc_kwargs, **kwargs)
    new_lifetime = np.matmul(
        np.multiply(np.asarray(x1), x1.intensity, out=None, **ufunc_kwargs, **kwargs),
        np.multiply(np.asarray(x2), x2.intensity, out=None, **ufunc_kwargs, **kwargs),
        out = None,
        **ufunc_kwargs,
        **kwargs
    )

    new_lifetime = np.divide(new_lifetime, new_intensity, out=None, **ufunc_kwargs, **kwargs) # can't seem to do inplace for some reason?

    if out is None:
        params = {}
        if x1._equal_params(x2):
            params = x1._inheritance_dict
        
        return FlimTrace(
            new_lifetime,
            intensity = new_intensity,
            **params
        )
    
    if not isinstance(out, FlimTrace):
        return NotImplemented
    
    out[...] = new_lifetime
    out.intensity = new_intensity
    return out


@FlimTrace.implements_ufunc(np.divide, "__call__")
def divide_call_flimtrace(x1, x2, /, out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
    """ Cannot divide two FlimTraces """
    ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}
    return flimtrace_ufunc_call_pattern(np.divide, x1, x2, out=out, **ufunc_kwargs, **kwargs)

@FlimTrace.implements_ufunc(np.true_divide, "__call__")
def true_divide_call_flimtrace(x1, x2, /, out = None, where = True, casting = 'same_kind', order = 'K', dtype=None, **kwargs):
    """ Cannot divide two FlimTraces """
    ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}
    return flimtrace_ufunc_call_pattern(np.true_divide, x1, x2, out=out, **ufunc_kwargs, **kwargs)

@FlimTrace.implements_ufunc(np.negative, "__call__")
def negative_call_flimtrace(x, / , out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
    if out is None:
        return FlimTrace(
            x.__array__(),
            intensity = np.negative(x.intensity, out=None, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
        )
    if not isinstance(out, FlimTrace):
        logging.warning(
            "Calling numpy {np.negative.__name__} on a FlimTrace but specifying a return value that is not a FlimTrace. "
            "Performing standard numpy {np.negative.__name__} on the intensity data.",
            stacklevel=2
        )
        return np.negative(x.intensity, out = out, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
    
    x : FlimTrace
    out[...] = x.lifetime # copy
    out.intensity = np.negative(x.intensity)
    return out

@FlimTrace.implements_ufunc(np.positive, "__call__")
def positive_call_flimtrace(x, / , out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
    if out is None:
        return FlimTrace(
            x.__array__(),
            intensity = np.positive(x.intensity, out=None, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
        )
    if not isinstance(out, FlimTrace):
        logging.warning(
            "Calling numpy {np.positive.__name__} on a FlimTrace but specifying a return value that is not a FlimTrace. "
            "Performing standard numpy {np.positive.__name__} on the intensity data.",
            stacklevel=2
        )
        return np.positive(x.intensity, out = out, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
    
    x : FlimTrace
    out[...] = x.lifetime # copy
    out.intensity = np.positive(x.intensity)
    return out

### REDUCE UFUNC #######

def flimtrace_ufunc_reduce_pattern(numpy_ufunc : np.ufunc, array, axis=0, dtype=None, out=None, keepdims=False, initial=None, where=True):
    """
    A pattern for ufunc reduction on FlimTrace data that do not perform addition-like phenomena
    
    The general procedure is to apply the ufunc to the intensity and return a FlimTrace with unchanged FLIM data
    """
    if not isinstance(array, FlimTrace):
        return numpy_ufunc.reduce(array, axis=axis, dtype = dtype, out=out, keepdims = False, initial=initial, where = where)
    
    if out is None:    
        return FlimTrace(
            array.__array__(),
            intensity = numpy_ufunc.reduce(array, axis=axis, dtype = dtype, out=None, keepdims = False, initial=initial, where = where),
            **array._inheritance_dict
        )
    if not isinstance(out, FlimTrace):
        logging.warning(
            "Calling numpy {np_ufunc.__name__}.reduce on a FlimTrace but specifying a return value that is not a FlimTrace. "
            "Performing standard numpy {np_ufunc.__name__}.reduce on the intensity data.",
            stacklevel=2
        )
        return numpy_ufunc.reduce(array, axis=axis, dtype = dtype, out=out, keepdims = False, initial=initial, where = where)
    return numpy_ufunc.reduce(array.intensity, axis=axis, dtype = dtype, out=out.intensity, keepdims = False, initial=initial, where = where)

# #TODO WRITE THESE

@FlimTrace.implements_ufunc(np.add, "reduce")
def add_reduce_flimtrace(array : FlimTrace, axis = 0, dtype = None, out = None, keepdims = False, initial = None, where = True, **kwargs):
    """ Adds directly to the intensity if a non-FlimTrace is provided """

    ufunc_kwargs = {'axis' : axis, 'dtype' : dtype, 'keepdims' : keepdims, 'initial' : initial, 'where' : where}

    new_intensity = np.add.reduce(array.intensity, out=None, **ufunc_kwargs, **kwargs)
    new_lifetime = np.add.reduce(
        np.multiply(np.asarray(array), array.intensity, out=None, **ufunc_kwargs, **kwargs),
        out = None,
        **ufunc_kwargs,
        **kwargs
    )

    new_lifetime = np.divide(new_lifetime, new_intensity, out=None, **ufunc_kwargs, **kwargs) # can't seem to do inplace for some reason?

    if out is None:
        params = array._inheritance_dict
        
        return FlimTrace(
            new_lifetime,
            intensity = new_intensity,
            **params
        )
    
    if not isinstance(out, FlimTrace):
        return NotImplemented
    
    out[...] = new_lifetime
    out.intensity = new_intensity
    return out

@FlimTrace.implements_ufunc(np.subtract, "reduce")
def subtract_reduce_flimtrace(array : FlimTrace, axis = 0, dtype = None, out = None, keepdims = False, initial = None, where = True, **kwargs):
    """ Haven't decided if this implementation actually makes sense! """
    ufunc_kwargs = {'axis' : axis, 'dtype' : dtype, 'keepdims' : keepdims, 'initial' : initial, 'where' : where}

    new_intensity = np.subtract.reduce(array.intensity, out=None, **ufunc_kwargs, **kwargs)
    new_lifetime = np.subtract.reduce(
        np.multiply(np.asarray(array), array.intensity, out=None, **ufunc_kwargs, **kwargs),
        out = None,
        **ufunc_kwargs,
        **kwargs
    )

    new_lifetime = np.divide(new_lifetime, new_intensity, out=None, **ufunc_kwargs, **kwargs) # can't seem to do inplace for some reason?

    if out is None:
        params = array._inheritance_dict
        
        return FlimTrace(
            new_lifetime,
            intensity = new_intensity,
            **params
        )
    
    if not isinstance(out, FlimTrace):
        return NotImplemented
    
    out[...] = new_lifetime
    out.intensity = new_intensity
    return out

#@FlimTrace.implements_ufunc(np.multiply, "reduce")
#def multiply_reduce_flimtrace(array : FlimTrace, axis = 0, dtype = None, out = None, keepdims = False, initial = None, where = True, **kwargs):
#    """ Cannot multiply two FlimTraces, should I even implement this method? """
#
#    ufunc_kwargs = {'axis' : axis, 'dtype' : dtype, 'keepdims' : keepdims, 'initial' : initial, 'where' : where}
#    return flimtrace_ufunc_reduce_pattern(np.multiply, array, out=out, **ufunc_kwargs, **kwargs)

# @FlimTrace.implements_ufunc(np.matmul, "reduce")
# def matmul_reduce_flimtrace(x1, x2, /, out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):

#     ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}
#     retval = flimtrace_ufunc_call_pattern(np.multiply, x1, x2, out=out, **ufunc_kwargs, **kwargs)
    
#     if not (retval is NotImplemented):
#         return retval

#     # Both are FlimTraces! Now we can perform the matrix multiplication, which is an array of sums,
#     # appropriately, inheriting the lifetime values that accompany performing such a sum.

#     x1 : FlimTrace
#     x2 : FlimTrace

#     new_intensity = np.matmul(x1.intensity, x2.intensity, out=None, **ufunc_kwargs, **kwargs)
#     new_lifetime = np.matmul(
#         np.multiply(np.asarray(x1), x1.intensity, out=None, **ufunc_kwargs, **kwargs),
#         np.multiply(np.asarray(x2), x2.intensity, out=None, **ufunc_kwargs, **kwargs),
#         out = None,
#         **ufunc_kwargs,
#         **kwargs
#     )

#     new_lifetime = np.divide(new_lifetime, new_intensity, out=None, **ufunc_kwargs, **kwargs) # can't seem to do inplace for some reason?

#     if out is None:
#         params = {}
#         if x1._equal_params(x2):
#             params = x1._inheritance_dict
        
#         return FlimTrace(
#             new_lifetime,
#             intensity = new_intensity,
#             **params
#         )
    
#     if not isinstance(out, FlimTrace):
#         return NotImplemented
    
#     out[...] = new_lifetime
#     out.intensity = new_intensity
#     return out


# @FlimTrace.implements_ufunc(np.divide, "reduce")
# def divide_reduce_flimtrace(x1, x2, /, out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
#     """ Cannot divide two FlimTraces """
#     ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}
#     return flimtrace_ufunc_call_pattern(np.divide, x1, x2, out=out, **ufunc_kwargs, **kwargs)

# @FlimTrace.implements_ufunc(np.true_divide, "reduce")
# def true_divide_reduce_flimtrace(x1, x2, /, out = None, where = True, casting = 'same_kind', order = 'K', dtype=None, **kwargs):
#     """ Cannot divide two FlimTraces """
#     ufunc_kwargs = {'where' : where, 'casting' : casting, 'order' : order, 'dtype' : dtype}
#     return flimtrace_ufunc_call_pattern(np.true_divide, x1, x2, out=out, **ufunc_kwargs, **kwargs)

# @FlimTrace.implements_ufunc(np.negative, "reduce")
# def negative_reduce_flimtrace(x, / , out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
#     if out is None:
#         return FlimTrace(
#             x.__array__(),
#             intensity = np.negative(x.intensity, out=None, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
#         )
#     if not isinstance(out, FlimTrace):
#         logging.warning(
#             "Calling numpy {np.negative.__name__} on a FlimTrace but specifying a return value that is not a FlimTrace. "
#             "Performing standard numpy {np.negative.__name__} on the intensity data.",
#             stacklevel=2
#         )
#         return np.negative(x.intensity, out = out, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
    
#     x : FlimTrace
#     out[...] = x.lifetime # copy
#     out.intensity = np.negative(x.intensity)
#     return out

# @FlimTrace.implements_ufunc(np.positive, "reduce")
# def positive_reduce_flimtrace(x, / , out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
#     if out is None:
#         return FlimTrace(
#             x.__array__(),
#             intensity = np.positive(x.intensity, out=None, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
#         )
#     if not isinstance(out, FlimTrace):
#         logging.warning(
#             "Calling numpy {np.positive.__name__} on a FlimTrace but specifying a return value that is not a FlimTrace. "
#             "Performing standard numpy {np.positive.__name__} on the intensity data.",
#             stacklevel=2
#         )
#         return np.positive(x.intensity, out = out, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
    
#     x : FlimTrace
#     out[...] = x.lifetime # copy
#     out.intensity = np.positive(x.intensity)
#     return out