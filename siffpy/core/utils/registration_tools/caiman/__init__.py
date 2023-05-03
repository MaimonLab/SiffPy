from typing import TYPE_CHECKING

import numpy as np

from siffreadermodule import SiffIO
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationInfo, RegistrationType
)
#from siffpy.core.utils.registration_tools.siffpy.alignment import (
#    align_to_reference
#)
#rom siffpy.core.utils.registration_tools.siffpy.registration_method import (
#    register_frames
#)

if TYPE_CHECKING:
    from siffpy.core.utils import ImParams
    from siffreadermodule import SiffIO

class CaimanRegistrationInfo(RegistrationInfo):

    backend = RegistrationType.Caiman

    def __init__(self, siffio : 'SiffIO', im_params : 'ImParams'):
        super().__init__(siffio, im_params)

    def register(self, *args, **kwargs):
        raise NotImplementedError()
        #return super().register(*args, **kwargs)

    def align_to_reference(
            self,
            image : np.ndarray,
            z_plane : int,
            *args, **kwargs
        )->tuple[int,int]:
        raise NotImplementedError()