import logging
from typing import Callable, Iterable

import numpy as np
from dask import delayed
import dask.array as da

from ...siffpy import SiffReader
from .napari_interface import NapariInterface

class FrameViewer(NapariInterface):
    """
    Gives access to a napari Viewer to
    create a window with settings optimized
    for interfacing with individual frames
    of a .siff file.

    Behaves very much LIKE a subclass without
    actually BEING a subclass. All attributes
    accessed (methods, etc.) first check if they
    are part of the object itself, then 

    Accepts all napari.Viewer args and kwargs
    in addition to custom kwargs as follows:

    Arguments
    ---------

    siffreader : siffpy.SiffReader

        A SiffReader object linekd to an open
        .siff or .tiff file.

    Keyword arguments
    -----------------

    load_frames : bool (default is False)

        Whether to pre-load all of the frames in the
        array. If False, frames will be loaded lazily
        using Dask. Using Dask means that during execution
        it will be slower to view new frames, but you won't
        have to wait (sometimes a very long time!) to load
        every single frame into memory before the images
        are loaded into numpy arrays

    batch_fcn : Callable

        What function to call to generate each batch.
        If left as None, it will simply produce
        a each volume. Does nothing if load_frames is True.
        If batch_iter is provided, then batch_fcn will
        be called on each one of batch_iter when demanded

    batch_iter : Iterable

        If a custom batch_fcn is provided, then it
        either must operate on each volume or a
        custom batch_iter must be provided to iterate
        through to produce each successive batch.

    Attributes
    ----------

    siffreader : siffpy.SiffReader

        A pointer to the same SiffReader object provided
        upon initialization

    """

    def __init__(
            self,
            siffreader : SiffReader,
            *args,
            load_frames : bool = False,
            batch_fcn : Callable = None,
            batch_iter : Iterable = None,
            image_opts : dict = None,
            **kwargs
        ):

        super().__init__(siffreader, *args, **kwargs, title = 'Frame viewer')
        # dumb bug prevents setting this in the init
        self.viewer.dims.axis_labels = siffreader.im_params.axis_labels

        if load_frames:
            logging.warn("Loading all frames. This might take a while...\n")
            
            stack = np.array(siffreader.get_frames(
                frames=list(range(siffreader.num_frames)),
                registration_dict= siffreader.registrationDict
            )).reshape(siffreader.im_params.stack)
        
        else:
            # uses dask to load volumes at a time.
            if batch_iter is None: # batch_iter is an iterable describing each batch
                if not batch_fcn is None:
                    logging.warn(
                        """"
                        \n\n\t
                        Using a custom function for
                        viewing frames, but did not provide
                        a custom iterable. Default to
                        iterating through volumes -- make
                        sure this is the intended behavior!
                        \n\n
                        """
                    )
                # each batch is a volume, this iter says which volume to grab
                batch_iter = range(siffreader.im_params.num_volumes)
            
            # function to get with each output of batch_iter
            if batch_fcn is None:
                def volume_get(volume_idx):
                    # Local function executed over and over
                    frame_start = volume_idx * siffreader.im_params.frames_per_volume
                    return np.array(
                        siffreader.get_frames(
                            frames=list(range(frame_start,frame_start+siffreader.im_params.frames_per_volume)),
                            registration_dict=siffreader.registration_dict
                        )
                    ).reshape(siffreader.im_params.volume)
                batch_fcn = volume_get
            
            dask_delayed_reads = [delayed(batch_fcn)(sample) for sample in batch_iter]

            dask_ars = [
                da.from_delayed(reader, shape = siffreader.im_params.volume, dtype=np.uint16)
                for reader in dask_delayed_reads
            ]

            stack = da.stack(dask_ars, axis =0)
        
        contrast = None
        if not image_opts is None:
            if 'clim' in image_opts:
                contrast = image_opts['clim']
                if not any(contrast):
                    contrast = None

        channel_axis = None
        if siffreader.im_params.num_colors > 1:
            channel_axis = 2

        self.add_image(
            data=stack,
            name='Raw images (one FOV)',
            scale = siffreader.im_params.scale,
            multiscale=False,
            channel_axis=channel_axis,
            contrast_limits = contrast,
            rgb = False
            )