"""
Wrapper code to facilitate running suite2p from
inside siffpy
"""
from typing import TYPE_CHECKING

try:
    from suite2p import default_ops
    #from suite2p.registration import registration_wrapper
except ImportError:
    raise ImportError(
        "Suite2p is not installed. Please install suite2p to use this module."
    )

import numpy as np

from siffreadermodule import SiffIO
from siffpy.core.utils import ImParams
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationInfo, RegistrationType
)


class Suite2pRegistrationInfo(RegistrationInfo):

    backend : RegistrationType = RegistrationType.Suite2p

    def __init__(self, siffio : 'SiffIO', im_params : 'ImParams'):
        super().__init__(siffio, im_params)

    def register(self, *args, **kwargs):
        ops = default_ops()
        if 'ops' in kwargs:
            ops = {**ops, **kwargs['ops']}
            del kwargs['ops']

        #input_frames = self.siffio.get_frames(frames=[])
        #out_frames = np.zeros(self.im_params.array_shape)
        #out_tuple = registration_wrapper(
        #    *args, **kwargs
        #)

       # regd = registration_wrapper(
    #outputs, f_raw = frames.astype(np.int16),
    #ops=
    #{
    #    **default_ops(),
    #    'nchannels': 1,
    #    'functional_chan': 1,
    #    'do_bidiphase': True,
    #    'bidi_corrected': True,
    #    'nonrigid' : False,
    #    'nplanes' : sr.im_params.num_slices,
    #    'batch_size' : 200,
    #    'nimg_init' : 100,
    #}
#)
        raise NotImplementedError()

    def align_to_reference(
            self,
            image : np.ndarray,
            z_plane : int
        )->tuple[int,int]:
        raise NotImplementedError()