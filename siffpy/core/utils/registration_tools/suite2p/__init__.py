"""
Wrapper code to facilitate running suite2p from
inside siffpy
"""
from inspect import Parameter
from typing import Tuple, Dict, TYPE_CHECKING
import warnings

import numpy as np

#from siffreadermodule import SiffIO
#from corrosiffpy import SiffIO
from siffpy.core.utils import ImParams
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationInfo, RegistrationType, populate_dict_across_colors
)

if TYPE_CHECKING:
    from corrosiffpy import SiffIO

suite2p_loaded = False
try:
    from suite2p import default_ops
    suite2p_loaded = True
except ImportError:
    warnings.warn(
        "Suite2p is not installed. Please install suite2p if you "
        + "intend to use the `suite2p` registration method."
    )
except ValueError:
    warnings.warn(
        "Suite2p is NOT compatible with Python >=3.11 due to bad "
        + "type annotations. Please use Python 3.10 or lower if you "
        + "want to call registration methods."
    )

    pass

SUITE2P_OPS = [
    'batch_size', 'do_bidiphase', 'smooth_sigma',
    'maxregshift', 'smooth_sigma_time', 'norm_frames',
    'nonrigid', 'two_step_registration', 'nimg_init',
]

def sct_defaults(suite2p_default_ops : Dict)->Dict:
    """ TODO: Make this settable! Read from a file?? """
    suite2p_default_ops['do_bidiphase'] = True
    suite2p_default_ops['nonrigid'] = False
    suite2p_default_ops['maxregshift'] = 0.4
    suite2p_default_ops['two_step_registration'] = True
    suite2p_default_ops['smooth_sigma'] = 2.0
    #suite2p_default_ops['batch_size'] = 300
    suite2p_default_ops['two_step_registration'] = False
    suite2p_default_ops['nimg_init'] = 300
    #suite2p_default_ops['norm_frames'] = F
    return suite2p_default_ops

class Suite2pRegistrationInfo(RegistrationInfo):
    
    multithreading_compatible = False
    backend : RegistrationType = RegistrationType.Suite2p

    saved_attrs = [
        '_ops',
    ]

    if suite2p_loaded:
        registration_params = {
            **{
                str(key) : Parameter(
                    str(key),
                    Parameter.KEYWORD_ONLY,
                    default=val,
                    annotation= type(val)
                )
                for key, val in sct_defaults(default_ops()).items()
                if key in SUITE2P_OPS
            },
            'align_by_chan2' : Parameter(
                'align_by_chan2',
                Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool
            )
        }

    def __init__(self, siffio : 'SiffIO', im_params : 'ImParams'):
        super().__init__(siffio, im_params)

    def register(self,
        siffio : 'SiffIO',
        *args,
        alignment_color_channel : int = 0,
        **kwargs
        ):
        """
        Registers individual planes using suite2p's registration method.
        
        If a kwarg called `ops` is provided, that's passed to suite2p's
        registration_wrapper function. Otherwise, the default_ops are used.
        """

        try:
            from suite2p import default_ops
            from suite2p.registration import register
        except ImportError:
            raise ImportError(
                "Suite2p is not installed. Please install suite2p to use this module."
            )


        frames = siffio.get_frames(
            frames = self.im_params.flatten_by_timepoints(color_channel=alignment_color_channel),
            registration = {}, # guarantee the raw frames
        ).astype(np.float32).reshape(-1, *self.im_params.volume_one_color)

        registered_frames = np.zeros_like(frames)

        ops = {
            **default_ops(),    
            **kwargs,
        }

        # Each list element is a tuple:
        # reference image, _, _, _, offsets (y, x), _, _
        reg_rets = [ # hee hee
            register.registration_wrapper(
                registered_frames[: , k, :, :].squeeze(),
                # scale f_raw by 100 since suite2p averages and THEN casts to uint16,
                # this keeps the values from being truncated to 0
                f_raw = 100*frames[:, k, :, :].squeeze(),
                ops = ops
            )
            for k in range(self.im_params.num_slices)
        ]

        self.reference_frames = np.array(
            [reg_ret[0] for reg_ret in reg_rets]
        ).astype(np.float32)/100

        frame_idxs = self.im_params.framelist_by_slice(color_channel = alignment_color_channel)

        self.yx_shifts = {}
        ysize, xsize = self.im_params.ysize, self.im_params.xsize
        for registration, framelist in zip(reg_rets, frame_idxs): # iterate over slices
            y_offsets = -registration[4][0]
            x_offsets = -registration[4][1]
            offsets = np.array([y_offsets, x_offsets]).T
            for frame_idx, offset in zip(framelist, offsets): # iterate over frames in slice
                self.yx_shifts[frame_idx] = (int(offset[0]) % ysize, int(offset[1]) % xsize)

        populate_dict_across_colors(
            self.im_params,
            alignment_color_channel,
            self.yx_shifts
        )

        self.registration_color_channel = alignment_color_channel
        self._ops = ops

    def align_to_reference(
        self,
        image : np.ndarray,
        z_plane : int
        )->Tuple[int,int]:
        raise NotImplementedError()