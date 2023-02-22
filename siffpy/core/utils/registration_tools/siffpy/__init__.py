import numpy as np

from siffreadermodule import SiffIO
from siffpy.core.utils import ImParams
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
    """
    A lightweight and simple version of suite2p's registration
    method. Benefits from being able to use the SiffIO object,
    rather than needing all of the frames provided as a numpy
    array up front. For very large arrays, this is very helpful.
    For small ones, you're likely better off using suite2p's
    registration method.
    """

    def __init__(self, siffio : SiffIO, im_params : ImParams):
        super().__init__(siffio, im_params, RegistrationType.Siffpy)

    def register(self, siffio, *args, num_cycles : int = 2, **kwargs):
        """
        Registers using the siffpy registration method, which
        is fairly similar to suite2p's registration method.
        Overview of the algorithm:

            - Sample a subset of frames, take their average
            - Take another subset of frames, find those most
            correlated to the average, and take their average
            to use as a reference.
            - Align all frames to these references by finding
            the location of the maximum of the ratio of their
            FFT to the reference image's FFT.

            Each successive cycle takes the preceding one's
            final alignment shifts as the starting point for
            creating a new reference image (from which the
            next batch of most-correlated frames is produced).

        Keyword args are passed to
        `siffpy.core.utils.registration_tools.siffpy.registration_method.register_frames`.

        Parameters
        ---------

        num_cycles : int

            The number of cycles to run registration. This is
            how many times to run all of the steps enumerated
            above.

        """
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
    
    def align_reference_frames(self):
        raise NotImplementedError()