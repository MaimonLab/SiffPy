"""
Wrapper code to facilitate running suite2p from
inside siffpy
"""
from inspect import Parameter
from typing import Tuple, Dict, TYPE_CHECKING, Optional
import warnings

import numpy as np

#from siffreadermodule import SiffIO
#from corrosiffpy import SiffIO
from siffpy.core.utils import ImParams
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationInfo, RegistrationType, populate_dict_across_colors,
    MROIRegistrationInfo
)

if TYPE_CHECKING:
    from corrosiffpy import SiffIO
    from siffpy.core.utils.im_params.im_params import ImagingROI

suite2p_loaded = False
try:
    import suite2p
    from suite2p import default_ops
    # New `suite2p` makes default_ops a separate module,
    # has somewhat new settings, urgh...
    if hasattr(suite2p, 'default_settings'):
        # from suite2p import default_settings
        # from suite2p.default_ops import default_ops as _default_ops
        # def default_ops():
        #     return {**_default_ops(), **default_settings()}
        settings_kwarg = 'settings'
    else:
        settings_kwarg = 'ops'
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
    suite2p_default_ops['smooth_sigma_time'] = 0
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

    @classmethod
    def mroi_class(cls) -> type[MROIRegistrationInfo]:
        return MROISuite2pRegistrationInfo

    def register(self,
        siffio : 'SiffIO',
        *,
        alignment_color_channel : int = 0,
        z_align : bool = False,
        volume_bounds : Optional[Tuple[int,int]] = None,
        planes_sequentially : bool = True,
        **kwargs
        ):
        """
        Registers individual planes using suite2p's registration method.

        # Arguments

        - siffio : SiffIO
            The SiffIO object to use for registration

        - alignment_color_channel : int
            The color channel to use for registration

        - z_align : bool
            Whether or not to align the planes to one another after
            registering each plane individually.

        - volume_bounds : Optional[Tuple[int,int]]
            Tuple of (start_volume, end_volume) to use for registration.
            If None, uses all volumes. Be careful with this one -- an incomplete
            registration dictionary will confuse analyses that try to read in
            all frames using the registration dictionary (missing frame keys result
            in errors thrown by the `SiffIO` class methods).

        - planes_sequentially : bool
            Whether to load and register each plane sequentially to save memory.
            I think this should be basically net zero in terms of speed, but it
            saves a lot of memory for large datasets (and probably saves time then too!)

        If `z_registration` is `True`, aligns the planes to one another by
        picking the least variable plane across the stack and fixing all other
        planes to that one.
        
        If a kwarg called `ops` is provided, that's passed to suite2p's
        registration_wrapper function. Otherwise, the default_ops are used.
        """

        try:
            #from suite2p import default_ops
            from suite2p.registration import register
        except ImportError:
            raise ImportError(
                "Suite2p is not installed. Please install suite2p to use this module."
            )

        # Planes sequentially saves memory by loading each plane into memory
        # in separate iterations
        if planes_sequentially:
            # raise NotImplementedError("Sequential plane registration not yet implemented.")

            reg_rets = []
            for plane_idx in range(self.im_params.num_slices):

                frames = siffio.get_frames(
                    frames = self.im_params.flatten_by_timepoints(
                        color_channel=alignment_color_channel,
                        reference_z=plane_idx,
                    ),
                    registration = {}, # guarantee the raw frames
                ).astype(np.float32).reshape(-1, *list(self.im_params.volume_one_color)[1:])

                registered_frames = np.zeros_like(frames)

                ops = {
                    **default_ops(),
                    **kwargs,
                }

                t_bounds = slice(None) if volume_bounds is None else slice(*volume_bounds)

                reg_rets.append(
                    register.registration_wrapper(
                        f_reg = registered_frames[t_bounds].squeeze(),
                        # scale f_raw by 100 since suite2p averages and THEN casts to uint16,
                        # this keeps the values from being truncated to 0
                        f_raw = 100*frames[t_bounds].squeeze(),
                        **{settings_kwarg : ops},
                        # ops = ops
                    )
                )
        
        # Is there any reason to do it this way still? I think it's useful if
        # suite2p develops 3d registration (maybe I should try suite3d?)
        else:
            frames = siffio.get_frames(
                frames = self.im_params.flatten_by_timepoints(color_channel=alignment_color_channel),
                registration = {}, # guarantee the raw frames
            ).astype(np.float32).reshape(-1, *self.im_params.volume_one_color)

            registered_frames = np.zeros_like(frames)

            ops = {
                **default_ops(),    
                **kwargs,
            }

            t_bounds = slice(None) if volume_bounds is None else slice(*volume_bounds)

            # Each list element is a tuple:
            # reference image, _, _, _, offsets (y, x), _, _
            reg_rets = [ # hee hee
                register.registration_wrapper(
                    f_reg = registered_frames[t_bounds, k, :, :].squeeze(),
                    # scale f_raw by 100 since suite2p averages and THEN casts to uint16,
                    # this keeps the values from being truncated to 0
                    f_raw = 100*frames[t_bounds, k, :, :].squeeze(),
                    **{settings_kwarg : ops},
                    # ops = ops
                )
                for k in range(self.im_params.num_slices)
            ]

        self.reference_frames = np.array(
            [reg_ret[0] for reg_ret in reg_rets]
        ).astype(np.float32)/100

        # align the reference frames to one another using the corrXY output of `registration_wrapper`:
        if z_align:
            corrXY_across_z = np.array([reg_ret[4][2] for reg_ret in reg_rets])
            raise NotImplementedError("Z-alignment not yet implemented")
        else:
            frame_idxs = self.im_params.framelist_by_slice(color_channel = alignment_color_channel)

            self.yx_shifts = {}
            ysize, xsize = self.im_params.ysize, self.im_params.xsize
            for registration, framelist in zip(reg_rets, frame_idxs): # iterate over slices
                framelist = framelist[t_bounds]
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
    
class MROISuite2pRegistrationInfo(MROIRegistrationInfo, Suite2pRegistrationInfo, ):
    """
    A class to store registration information for MROI data registered using suite2p.
    """

    default_class = Suite2pRegistrationInfo

    def register_mroi(
        self,
        siffio : 'SiffIO',
        roi : 'ImagingROI',
        *,
        alignment_color_channel : int = 0,
        z_align : bool = False,
        volume_bounds : Optional[Tuple[int,int]] = None,
        planes_sequentially : bool = True,
        **kwargs
        ):
        """
        Registers individual planes using suite2p's registration method
        on just the mask corresponding to the ROI.

        # Arguments

        - siffio : SiffIO
            The SiffIO object to use for registration

        - roi : ImagingROI
            The ROI to use for registration. Only the pixels within the mask of this ROI
            will be used for registration.

        - alignment_color_channel : int
            The color channel to use for registration

        - z_align : bool
            Whether or not to align the planes to one another after
            registering each plane individually.

        - volume_bounds : Optional[Tuple[int,int]]
            Tuple of (start_volume, end_volume) to use for registration.
            If None, uses all volumes. Be careful with this one -- an incomplete
            registration dictionary will confuse analyses that try to read in
            all frames using the registration dictionary (missing frame keys result
            in errors thrown by the `SiffIO` class methods).

        - planes_sequentially : bool
            Whether to load and register each plane sequentially to save memory.
            I think this should be basically net zero in terms of speed, but it
            saves a lot of memory for large datasets (and probably saves time then too!)

        If `z_registration` is `True`, aligns the planes to one another by
        picking the least variable plane across the stack and fixing all other
        planes to that one.
        
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

        # Planes sequentially saves memory by loading each plane into memory
        # in separate iterations
        if planes_sequentially:
            # raise NotImplementedError("Sequential plane registration not yet implemented.")

            reg_rets = []
            
            for plane_idx in range(self.im_params.num_slices):
                # plane_mask = np.zeros_like(roi.mask)
                # plane_mask[plane_idx] = roi.mask[plane_idx]

                frames = roi.from_masked(
                    siffio.get_roi_1d(
                        mask = roi.mask[plane_idx],
                        frames = self.im_params.flatten_by_timepoints(
                            color_channel=alignment_color_channel,
                            reference_z=plane_idx,
                        ),
                        registration = {}, # guarantee the raw frames
                    ).astype(np.float32),
                    to_minimal_image=True,
                    plane = plane_idx,
                )

                registered_frames = np.zeros_like(frames)

                ops = {
                    **default_ops(),
                    **kwargs,
                }

                t_bounds = slice(None) if volume_bounds is None else slice(*volume_bounds)

                reg_rets.append(
                    register.registration_wrapper(
                        f_reg = registered_frames[t_bounds].squeeze(),
                        # scale f_raw by 100 since suite2p averages and THEN casts to uint16,
                        # this keeps the values from being truncated to 0
                        f_raw = 100*frames[t_bounds].squeeze(),
                        # ops = ops
                        **{settings_kwarg : ops},
                    )
                )
        
        # Is there any reason to do it this way still? I think it's useful if
        # suite2p develops 3d registration (maybe I should try suite3d?)
        else:
            raise NotImplementedError(
                "Non-sequential plane registration not yet implemented for MROI data."
            )

        self.reference_frames = np.array(
            [reg_ret[0] for reg_ret in reg_rets]
        ).astype(np.float32)/100

        # align the reference frames to one another using the corrXY output of `registration_wrapper`:
        if z_align:
            corrXY_across_z = np.array([reg_ret[4][2] for reg_ret in reg_rets])
            raise NotImplementedError("Z-alignment not yet implemented")
        else:
            frame_idxs = self.im_params.framelist_by_slice(color_channel = alignment_color_channel)

            self.yx_shifts = {}
            
            # WARNING THIS DOESN'T WORK FOR NON-CONSISTENT RECTANGLES ACROSS PLANES!
            # TODO REDO FOR MORE GENERAL USE CASE.
            # ysize, xsize = roi.minimal_mask.shape[-2], roi.minimal_mask.shape[-1]
            ysize, xsize = self.im_params.ysize, self.im_params.xsize
            for registration, framelist in zip(reg_rets, frame_idxs): # iterate over slices
                framelist = framelist[t_bounds]
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
