"""
Array functions for FlimTraces

To be added to whenever I encounter something strange.
"""

from functools import reduce
from operator import add
import numpy as np
from ..traces import FlimTrace

@FlimTrace.implements_func(np.sum)
def sum_flimtrace(flimtrace : FlimTrace, out = None, **kwargs):
    if not (out is None or isinstance(out, FlimTrace)):
        return NotImplemented
    product_array = flimtrace.lifetime * flimtrace.intensity
    intensity_sum = np.sum(flimtrace.intensity, **kwargs)
    if out is None:
        return FlimTrace(
            np.nansum(product_array, **kwargs)/intensity_sum,
            intensity = intensity_sum,
            **flimtrace._inheritance_dict
        )
    else:
        if not (out.shape == intensity_sum.shape):
            raise ValueError("Provided FlimTrace must have same shape as returned sum")
        out[...] = np.nansum(product_array, **kwargs)/intensity_sum
        out.intensity = intensity_sum
        return out

@FlimTrace.implements_func(np.nansum)
def nansum_flimtrace(flimtrace : FlimTrace, out = None, **kwargs):
    if not (out is None or isinstance(out, FlimTrace)):
        return NotImplemented

    product_array = flimtrace.lifetime * flimtrace.intensity
    intensity_sum = np.nansum(flimtrace.intensity, **kwargs)
    if out is None:
        return FlimTrace(
            np.nansum(product_array, **kwargs)/intensity_sum,
            intensity = intensity_sum,
            **flimtrace._inheritance_dict
        )
    else:
        if not (out.shape == intensity_sum.shape):
            raise ValueError("Provided FlimTrace must have same shape as returned sum")
        out[...] = np.nansum(product_array, **kwargs)/intensity_sum
        out.intensity = intensity_sum
        return out

@FlimTrace.implements_func(np.mean)
def mean_flimtrace(flimtrace : FlimTrace, out = None, **kwargs):
    if not (out is None or isinstance(out, FlimTrace)):
        return NotImplemented
    
    product_array = flimtrace.lifetime * flimtrace.intensity
    intensity_sum = np.sum(flimtrace.intensity, **kwargs)
    intensity_mean = np.nanmean(flimtrace.intensity, **kwargs)
    
    if out is None:
        return FlimTrace(
            np.nansum(product_array, **kwargs)/intensity_sum,
            intensity = intensity_mean,
            **flimtrace._inheritance_dict
        )
    
    else:
        if not (out.shape == intensity_sum.shape):
            raise ValueError("Provided FlimTrace must have same shape as returned array")
        out[...] = np.nansum(product_array, **kwargs)/intensity_sum
        out.intensity = intensity_mean
        return out

@FlimTrace.implements_func(np.nanmean)
def nanmean_flimtrace(flimtrace : FlimTrace, out = None, **kwargs):
    if not (out is None or isinstance(out, FlimTrace)):
        return NotImplemented
    
    product_array = flimtrace.lifetime * flimtrace.intensity
    intensity_sum = np.nansum(flimtrace.intensity, **kwargs)
    intensity_mean = np.nanmean(flimtrace.intensity, **kwargs)
    
    if out is None:
        return FlimTrace(
            np.nansum(product_array, **kwargs)/intensity_sum,
            intensity = intensity_mean,
            **flimtrace._inheritance_dict
        )
    
    else:
        if not (out.shape == intensity_sum.shape):
            raise ValueError("Provided FlimTrace must have same shape as returned array")
        out[...] = np.nansum(product_array, **kwargs)/intensity_sum
        out.intensity = intensity_mean
        return out

@FlimTrace.implements_func(np.concatenate)
def concatenate_flimtrace(concats, *args, out = None, **kwargs):
    if not (out is None or isinstance(out, FlimTrace)):
        return NotImplemented
    
    if not all(isinstance(x, FlimTrace) for x in concats):
        return NotImplemented
    
    flim_concat = np.concatenate([x.__array__() for x in concats], **kwargs)
    intensity_concat = np.concatenate([x.intensity for x in concats], **kwargs)
    confidence_concat = None

    if out is None:
        params = None
        method = None
        angle = None,
        info_string = None
        if all(x.FLIMParams == concats[0].FLIMParams for x in concats):
            params = concats[0].FLIMParams
        if all(x.method == concats[0].method for x in concats):
            method = concats[0].method
        if all(x.angle == concats[0].angle for x in concats):
            angle = concats[0].angle
        info_string = reduce(add, (str(x.info_string) + " " for x in concats))
        
        return FlimTrace(
            flim_concat,
            intensity = intensity_concat,
            confidence = None,
            FLIMParams = params,
            method = method,
            angle = angle,
            info_string = info_string
        )
    
    else:
        if not (out.shape == flim_concat.shape):
            raise ValueError("Provided FlimTrace must have same shape as returned array")
        out[...] = flim_concat
        out.intensity = intensity_concat
        return out

@FlimTrace.implements_func(np.reshape)
def reshape_flimtrace(flimtrace : FlimTrace, newshape, **kwargs):
    return FlimTrace(
        np.reshape(flimtrace.__array__(), newshape, **kwargs),
        np.reshape(flimtrace.intensity, newshape, **kwargs),
        **flimtrace._inheritance_dict
    )

@FlimTrace.implements_func(np.append)
def append_flimtrace(flimtrace : FlimTrace, values : FlimTrace, **kwargs):
    if not (isinstance(flimtrace, FlimTrace) and isinstance(values, FlimTrace)):
        return NotImplemented
    return FlimTrace(
        np.append(flimtrace.__array__(), values.__array__(), **kwargs),
        intensity = np.append(flimtrace.intensity, values.intensity, **kwargs),
        **flimtrace._inheritance_dict
    )

@FlimTrace.implements_func(np.convolve)
def convolve_flimtrace(a, v, mode = 'full')->FlimTrace:
    
    if (isinstance(a, FlimTrace) and isinstance(v, FlimTrace)):
        raise ValueError("Cannot convolve two FlimTraces. One must be a standard array.")
    if not isinstance(a, FlimTrace):
        a, v = v, a
    
    prod_trace = a.__array__() * a.intensity
    prod_trace = np.nan_to_num(prod_trace) # must be 0s for the convolution to behave well
    intensity_conv = np.convolve(a.intensity, v, mode)
    prod_conv = np.convolve(prod_trace, v, mode)
    # will be returned to nan where intensity is 0 by the division step
    return FlimTrace(
        prod_conv/intensity_conv,
        intensity = intensity_conv,
        **a._inheritance_dict
    )

@FlimTrace.implements_func(np.ravel)
def ravel_flimtrace(a, order = 'C')->FlimTrace:
    return FlimTrace(
        np.ravel(a.__array__(),order),
        np.ravel(a.intensity,order),
        **a._inheritance_dict
    )

@FlimTrace.implements_func(np.sort)
def sort_flimtrace(flimtrace, sortby = 'flim', **kwargs):
    """ SORTS BY FLIM VALUE NOT INTENSITY """
    if sortby == 'flim':
        idxs = np.argsort(flimtrace.__array__(), **kwargs)
    elif sortby == 'intensity':
        idxs = np.argsort(flimtrace.intensity, **kwargs)
    else:
        raise ValueError("Invalid sortby parameter for FlimTrace. Must be 'flim' or 'intensity'.")
    return FlimTrace(
        flimtrace.__array__()[idxs],
        flimtrace.intensity[idxs],
        **flimtrace._inheritance_dict
    )

@FlimTrace.implements_func(np.squeeze)
def squeeze_flimtrace(flimtrace, axis=None):
    return FlimTrace(
        np.squeeze(flimtrace.__array__(),axis),
        np.squeeze(flimtrace.intensity, axis)
        **flimtrace._inheritance_dict
    )

@FlimTrace.implements_func(np.transpose)
def tranpose_flimtrace(flimtrace, axes):
    return FlimTrace(
        np.transpose(flimtrace.__array__(), axes),
        np.transpose(flimtrace.intensity, axes),
        **flimtrace._inheritance_dict
    )