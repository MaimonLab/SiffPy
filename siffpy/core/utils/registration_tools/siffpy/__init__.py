import numpy as np

from siffreadermodule import SiffIO
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationInfo, RegistrationType
)
from siffpy.core.utils.registration_tools.siffpy.alignment import (
    align_to_reference
)
from siffpy.core.utils.registration_tools.siffpy.registration_method import (
    register_frames
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
            z_plane : int,
            *args, **kwargs
        )->tuple[int,int]:
        return align_to_reference(
            self.reference_frames[z_plane],
            image,
            *args, **kwargs
        )