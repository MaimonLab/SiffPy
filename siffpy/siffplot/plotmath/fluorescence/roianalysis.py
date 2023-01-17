from typing import Union, Callable
from functools import reduce
import operator

import numpy as np

from siffpy.core import SiffReader
from siffpy.siffplot.utils.exceptions import NoROIException
from siffpy.siffplot.roi_protocols.rois import ROI, subROI
from siffpy.siffmath import fluorescence, fluorescence_fcns

def compute_roi_timeseries(siffreader : SiffReader, 
    roi : Union[ROI, list[ROI]], *args,
    fluorescence_method : Union[str, Callable] = None,
    color_list : Union[list[int], int]= None,
    **kwargs) -> np.ndarray:
    """
    Takes an ROI object and returns a numpy.ndarray corresponding to fluorescence method supplied applied to
    the ROI. Additional args and kwargs provided are supplied to the fluorescence method itself.

    Arguments
    ---------

    siffreader : siffpy.SiffReader

        The file I/O class for .siff files

    roi : siffpy.siffplot.roi_protocols.rois.roi.ROI

        Any ROI subclass or list of ROIs

    fluorescence_method : str or callable (optional)

        Which method to use to compute the timeseries from the
        frame specifications. Accepts any public function defined in 
        siffmath.fluorescence. If no argument is provided, defaults
        to dF/F with F0 defined as the fifth percentile signal in
        each ROI. NOT Normalized from 0 to 1 by default! If a callable
        is used, computes fluroescence with that function instead (with
        the expectation that this function will be the transformation
        from raw pixel photon counts into some readout a la dF/F).

    color_list : list[int], int, or None

        The color_list parameter as passed to `siffpy.SiffReader`'s sum_roi method.
        In brief, if a list 

    *args and other kwargs provided are passed directly to the fluorescence_
    method argument along with the full intensity profile, as:
        fluorescence_method(intensity, *args, **kwargs)
    .

    Returns
    -------

    roi_timeseries : np.ndarray OR returned type of the fluorescence method

        Array of shape (number_of_timebins,) corresponding
        to the analysis specified on the region contained by the
        ROI provided. If the fluorescence_method provided returns
        another type (this should always return a numpy array or
        subclass), it will return that!
    """
    registration_dict = None

    if hasattr(siffreader, 'registration_dict'):
        registration_dict = siffreader.registration_dict

    if isinstance(roi, (list, tuple)):
        if not all( isinstance(x, ROI) for x in roi ):
            raise TypeError("Not all objects provided in list are of type `ROI`.")
        roi_trace = np.sum(
            [
                siffreader.sum_roi(
                    individual_roi,
                    color_list = color_list,
                    registration_dict = registration_dict
                )
                for individual_roi in roi
            ],
            axis=0
        )
    elif isinstance(roi, ROI):
        roi_trace = siffreader.sum_roi(
            roi,
            color_list = color_list,
            registration_dict = registration_dict
        )
    else:
        raise TypeError(f"Parameter `roi` must be of type `ROI` or a list of `ROI`s.")

    if fluorescence_method is None:
        fluorescence_method = fluorescence.dFoF         

    # Optional alternatives
    if not callable(fluorescence_method):
        if not fluorescence_method in fluorescence_fcns():
            raise ValueError(
                "Fluorescence extraction method must either be a callable or a string. Available " +
                "string options are functions defined in siffmath.fluorescence. Those are:" +
                reduce(operator.add, ["\n\n\t"+name for name in fluorescence_fcns()])
            )
        fluorescence_method = getattr(fluorescence,fluorescence_method)

    return fluorescence_method(roi_trace, *args, **kwargs).flatten()

def compute_vector_timeseries(siffreader : SiffReader, 
    roi : ROI,
    *args,
    fluorescence_method : Union[str,Callable] = None,
    color_list : Union[list[int],int] = None,
    **kwargs
    )-> np.ndarray:
    """
    Takes an roi ROI with subROIs and uses it to segment the data
    linked in the siffreader file into individual ROIs and
    return some analysis on each ROI. Does not store attribute
    vector_timeseries in thePlotter -- but many other functions
    that use this one do.

    Arguments
    ---------
    
    siffreader : siffpy.SiffReader

        The file I/O class for .siff files

    roi : siffpy.siffplot.roi_protocols.rois.roi.ROI

        Any ROI subclass that has a 'subROIs' attribute.

    fluorescence_method : str or callable (optional)

        Which method to use to compute the vector_timeseries from the
        frame specifications.
        
        Accepts a string naming any public function defined in 
        siffmath.fluorescence, even if its signature is not appropriate here.
        If no argument is provided, defaults
        to dF/F with F0 defined as the fifth percentile signal in
        each ROI.

        Any Callable provided must have a signature compliant with:
        method(intensity : np.ndarray, *args, **kwargs)

    args and other kwargs provided are passed directly to the method
    used to compute the vector_timeseries. None need to be provided
    to use the default functionality, however.

    Returns
    -------

    vector_timeseries : np.ndarray

        Array of shape (number_of_subROIs, number_of_timebins) corresponding
        to the analysis specified on each of the subROIs of the argument
        ROI provided.
    """
    if roi is None:
        raise NoROIException("No ROI provided to compute_vector_timeseries")
    
    if not hasattr(roi, 'subROIs'):
        raise NoROIException(f"Provided roi {roi} of type {type(roi)} does not have attribute 'subROIs'.")
    if not all(map(lambda x: isinstance(x, subROI), roi.subROIs)):
        raise NoROIException("Supposed subROIs (segments, columns, etc.) are not actually of type subROI. Presumed error in implementation.")

    return np.array(
        [
            compute_roi_timeseries(
                siffreader,
                sub_roi, *args,
                fluorescence_method = fluorescence_method,
                color_list = color_list,
                **kwargs,
            ) for sub_roi in roi.subROIs
        ]
    )