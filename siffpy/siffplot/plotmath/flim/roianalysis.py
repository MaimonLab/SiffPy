from typing import Union, Callable

import numpy as np

from ....core import SiffReader
from ...roi_protocols.rois import ROI, subROI
from ....siffutils import FLIMParams
from ....siffmath.flim import FlimTrace
from ...utils.exceptions import NoROIException

def compute_roi_timeseries(siffreader : SiffReader, roi : Union[ROI, list[ROI]],
    flim_params : Union[FLIMParams, list[FLIMParams]],
    color_list : Union[list[int], int]= None,
    ) -> np.ndarray:
    """
    Takes an ROI object and returns a FlimTrace
    corresponding to the
    empirical lifetime of the
    pooled pixels within the ROI.
    
    Arguments
    ---------

    flim_params : FLIMParams

        Params 

    roi : siffpy.siffplot.roi_protocols.rois.roi.ROI

        Any ROI subclass or list of ROIs

    color_list : list[int], int, or None

        The color_list parameter as passed to `siffpy.SiffReader`'s sum_roi method.
        In brief, if a list 

    Returns
    -------

    roi_timeseries : np.ndarray

        Array of shape (number_of_timebins,) corresponding
        to the analysis specified on the region contained by the
        ROI provided
    """
    registration_dict = None
    if hasattr(siffreader, 'registration_dict'):
        registration_dict = siffreader.registration_dict
 
    if isinstance(roi, (list, tuple)):
        if not all( isinstance(x, ROI) for x in roi ):
            raise TypeError("Not all objects provided in list are of type `ROI`.")
        roi_trace = np.sum(
            FlimTrace([
                siffreader.sum_roi_flim(
                    flim_params,
                    individual_roi,
                    color_list = color_list,
                    registration_dict = registration_dict
                )
                for individual_roi in roi
            ]),
            axis=0
        )
    elif isinstance(roi, ROI):
        roi_trace = siffreader.sum_roi_flim(
            flim_params,
            roi,
            color_list = color_list,
            registration_dict = registration_dict
        )
    else:
        raise TypeError(f"Parameter `roi` must be of type `ROI` or a list of `ROI`s.")

    return roi_trace
    
def compute_vector_timeseries(siffreader : SiffReader, 
    roi : ROI,
    flim_params : Union[list[FLIMParams],FLIMParams],
    color_list : Union[list[int],int] = None,
    )-> np.ndarray:
    """
    Takes an roi ROI with subROIs and uses it to segment the data
    linked in the siffreader file into individual ROIs and
    return the empirical lifetime of each ROI. 

    Arguments
    ---------

    flim_params : siffpy.siffutils.FLIMParams

    roi : siffpy.siffplot.roi_protocols.rois.roi.ROI

        Any ROI subclass that has a 'subROIs' attribute.

    color_list : list[int], int, or None

        The color_list parameter as passed to `siffpy.SiffReader`'s sum_roi method.
        In brief, if a list 

    Returns
    -------

    vector_timeseries : siffpy.siffmath.flim.FlimTrace

        Array of shape (number_of_subROIs, number_of_timebins) corresponding
        to the analysis specified on each of the subROIs of the argument
        ROI provided.
    """
    if not hasattr(roi, 'subROIs'):
        raise NoROIException(f"Provided roi {roi} of type {type(roi)} does not have attribute 'subROIs'.")
    if not all(map(lambda x: isinstance(x, subROI), roi.subROIs)):
        raise NoROIException("Supposed subROIs (segments, columns, etc.) are not actually of type subROI. Presumed error in implementation.")

    return FlimTrace(
        [
            compute_roi_timeseries(
                sub_roi,
                flim_params,
                color_list = color_list,
            )
            for sub_roi in roi.subROIs
        ]
    )