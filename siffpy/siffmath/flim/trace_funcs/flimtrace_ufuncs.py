"""
Array ufunctions for FlimTraces

To be added to whenever I encounter something strange.
"""
import logging

import numpy as np

from ..traces import FlimTrace

# REFERENCE OUTS just so I can remind myself how that looks
        # if outputs:
        #     out_args = []
        #     for j, output in enumerate(outputs):
        #         if isinstance(output, FluorescenceTrace):
        #             out_args.append(output.view(np.ndarray))
        #         else:
        #             out_args.append(output)
        #     kwargs['out'] = tuple(out_args)
        # else:
        #     outputs = (None,) * ufunc.nout

def flimtrace_ufunc_call_pattern(numpy_ufunc, *args, out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
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
        logging.warn(
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
    retval = flimtrace_ufunc_call_pattern(np.multiply, x1, x2, out=out, **ufunc_kwargs, **kwargs)
    
    if not (retval is NotImplemented):
        return retval

    # Both are FlimTraces, now we can do something actually interesting!
    x1 : FlimTrace
    x2 : FlimTrace

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
        logging.warn(
            "Calling numpy {np.negative.__name__} on a FlimTrace but specifying a return value that is not a FlimTrace. "
            "Performing standard numpy {np.negative.__name__} on the intensity data.",
            stacklevel=2
        )
        return np.negative(x.intensity, out = out, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
    
    x : FlimTrace
    out[...] = x.lifetime # copy
    out.intensity = np.negative(x.intensity)

@FlimTrace.implements_ufunc(np.positive, "__call__")
def positive_call_flimtrace(x, / , out = None, where = True, casting = 'same_kind', order = 'K', dtype = None, **kwargs):
    if out is None:
        return FlimTrace(
            x.__array__(),
            intensity = np.positive(x.intensity, out=None, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
        )
    if not isinstance(out, FlimTrace):
        logging.warn(
            "Calling numpy {np.positive.__name__} on a FlimTrace but specifying a return value that is not a FlimTrace. "
            "Performing standard numpy {np.positive.__name__} on the intensity data.",
            stacklevel=2
        )
        return np.positive(x.intensity, out = out, where = where, casting = casting, order = order, dtype = dtype, **kwargs)
    
    x : FlimTrace
    out[...] = x.lifetime # copy
    out.intensity = np.positive(x.intensity)

