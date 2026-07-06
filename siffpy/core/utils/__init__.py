import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from siffpy import SiffReader
from siffpy.core.utils.im_params.im_params import ImParams  # noqa: F401

class MROIError(Exception):
    """ Raised when calling functions for standard imaging when mROI
    imaging functionality was used for the image acquisition """

    def __init__(self,
                 message : str = "These data were acquired using mroi functionality."
                 " Use the equivalent mROI function instead"
        ):
        super().__init__(message)

def warn_for_mroi(reader : 'SiffReader', mroi_called : bool = False):
    """
    Warn the user that MROI is not yet implemented explicitly
    """
    if mroi_called:
        if not reader.im_params.RoiManager.mroiEnable:
            warnings.warn(
                'You have called an mROI function but these data do not use the mROI functionality. \
                Please check your data and function calls to make sure this is what you intended.'
            )
        return
    if reader.im_params.RoiManager.mroiEnable:
        if not mroi_called:
            warnings.warn(
                'You have called a non-mROI function but these data use the mROI functionality. \
                Please check your data and function calls to make sure this is what you intended.'
            )