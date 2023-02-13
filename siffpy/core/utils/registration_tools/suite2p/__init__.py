"""
Wrapper code to facilitate running suite2p from
inside siffpy
"""

try:
    from suite2p.registration import registration_wrapper
except ImportError:
    raise ImportError(
        "Suite2p is not installed. Please install suite2p to use this module."
    )

import numpy as np

from siffreadermodule import SiffIO
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationInfo, RegistrationType
)

class Suite2pRegistrationInfo(RegistrationInfo):

    def __init__(self, siffio : SiffIO):
        super().__init__(siffio, RegistrationType.Suite2p)

    def register(self, *args, **kwargs):
        raise NotImplementedError()

    def align_to_reference(
            self,
            image : np.ndarray,
            z_plane : int
        )->tuple[int,int]:
        raise NotImplementedError()