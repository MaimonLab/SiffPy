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

class CaimanRegistrationInfo(RegistrationInfo):

    def __init__(self, siffio : SiffIO):
        super().__init__(siffio, RegistrationType.Caiman)

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