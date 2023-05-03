import builtins
import logging
from typing import TYPE_CHECKING

import numpy as np

from siffreadermodule import SiffIO
from siffpy.core.utils import ImParams
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationInfo, RegistrationType
)
from siffpy.core.utils.registration_tools.siffpy.alignment import (
    align_to_reference, suite2p_reference
)
from siffpy.core.utils.registration_tools.siffpy.registration_method import (
    register_frames
)

if TYPE_CHECKING:
    try:
        from tqdm import tqdm
    except ImportError:
        pass

class SiffpyRegistrationInfo(RegistrationInfo):
    """
    A lightweight and simple version of suite2p's registration
    method. Benefits from being able to use the SiffIO object,
    rather than needing all of the frames provided as a numpy
    array up front. For very large arrays, this is very helpful.
    For small ones, you're likely better off using suite2p's
    registration method.
    """

    backend = RegistrationType.Siffpy

    def __init__(self, siffio : SiffIO, im_params : ImParams):
        super().__init__(siffio, im_params)

    def register(
            self,
            siffio,
            *args,
            alignment_color_channel : int = 0,
            num_cycles : int = 2,
            align_z : bool = False,
            **kwargs
        ):
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
        framelists = self.im_params.framelist_by_slice(
            color_channel=alignment_color_channel
        )

        self.reference_frames = np.zeros(
            self.im_params.single_channel_volume,
        ).squeeze()

        pbar = None if not "tqdm" in kwargs else kwargs["tqdm"]

        if not pbar is None:
            pbar : tqdm
            pbar.reset(total=len(framelists))
        for z_idx, z_plane_frames in enumerate(framelists):
            print(f"Registering z-plane {z_idx}")
            if pbar is not None:
                pbar.update(n=z_idx)
                #pbar.
            for cycle in range(num_cycles):

                self.reference_frames[z_idx,...] = suite2p_reference(
                    siffio,
                    z_plane_frames,
                    yx_shifts = self.yx_shifts,
                    **kwargs,
                )
                self.yx_shifts = {
                    **self.yx_shifts,
                    **register_frames(
                        siffio,
                        self.reference_frames[z_idx, ...],
                        z_plane_frames,
                        registration_dict=self.yx_shifts,
                        pbar = pbar,
                        **kwargs
                    )
                }

        if align_z:
            logging.warn(
                "align_z is not yet implemented for siffpy registration. Ignoring."
            )

        self.registration_color_channel = alignment_color_channel
        # Now register the other color channels
        # using their corresponding element from the
        # original color channel
        reference_frame_list = self.im_params.framelist_by_color(color_channel=alignment_color_channel)
        for color_matlab_int in self.im_params.color_list:
            color_channel = color_matlab_int - 1
            if color_channel != alignment_color_channel:
                this_color_framelist = self.im_params.framelist_by_color(
                    color_channel=color_channel
                )
                for this_frame, old_frame in zip(this_color_framelist, reference_frame_list):
                    self.yx_shifts[this_frame] = self.yx_shifts[old_frame]
                                        

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