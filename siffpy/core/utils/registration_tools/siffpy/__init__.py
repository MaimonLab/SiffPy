import numpy as np

from siffreadermodule import SiffIO
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationInfo, RegistrationType
)

class SiffpyRegistrationInfo(RegistrationInfo):

    def __init__(self, siffio : SiffIO):
        super().__init__(siffio, RegistrationType.Siffpy)

    def register(self, *args, **kwargs):
        raise NotImplementedError()
        #return super().register(*args, **kwargs)

    def align_to_reference(
            self,
            image : np.ndarray,
            z_plane : int
        )->tuple[int,int]:
        raise NotImplementedError()