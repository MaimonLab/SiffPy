import logging
from typing import Callable, Union, Iterable

import numpy as np

from siffpy.core import SiffReader, FLIMParams
from siffpy.siffplot.napari_viewers.frame_viewer import FrameViewer

class FlimViewer(FrameViewer):
    """
    Gives access to a napari Viewer to
    create a window with settings optimized
    for interfacing with individual frames
    of a .siff file as FLIM data.

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

    params : siffpy.FLIMParams or list of FLIMParams

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

    IMAGE_LAYER_NAME = "Fluorescence lifetime (one FOV)"
    WINDOW_TITLE     = "Flim frame viewer"

    def __init__(
            self,
            siffreader : SiffReader,
            params : Union[list[FLIMParams], FLIMParams],
            *args,
            image_opts : dict = None,
            **kwargs,
        ):

        if not isinstance(params, list):
            params = [params]
        if not image_opts is None:
            if not 'clim' in image_opts:
                image_opts['clim'] = (1.0, siffreader.im_params.num_bins)

        self.use_flim = True # starting parameter, I intend this to be changed with a button
        self.params = params
        self.POOL_WIDTH = 10
        if siffreader.im_params.num_colors > 1:
            self.warning_window(
                "Haven't yet implemented multicolor FlimViewer",
                NotImplementedError("Haven't yet implemented multi-color-channel FlimViewer!")
            )

        super().__init__(siffreader, *args, image_opts = image_opts, **kwargs)

    def _default_volume_get(self)->Callable:
        """
        Returns the default function for the batch function of the get_frames layer.

        To be implemented differently by other subclasses.
        """
        intensity_get = super()._default_volume_get()

        def volume_get(volume_idx):
            # Local function executed over and over

            if self.use_flim:
                return self.siffreader.flimmap_across_time(
                    self.params[0],
                    timepoint_start = volume_idx,
                    timepoint_end = volume_idx + self.POOL_WIDTH,
                    timespan = self.POOL_WIDTH,
                    registration=self.siffreader.registration_dict
                ).reshape(*self.siffreader.im_params.volume)
            else:
                return intensity_get(volume_idx)

        return volume_get

    def _default_batch_iter(self)->Iterable:
        return range(self.siffreader.im_params.num_volumes-self.POOL_WIDTH)

    def _default_preload_frames(self)->np.ndarray:
        """
        Returns all frames loaded at the initiation of the reader.
        """

        logging.warn("Loading all frames. This might take a while...\n")
        
        if self.use_flim:
            raise NotImplementedError("Haven't implemented for the FlimViewer yet.")
        else:
            return super()._default_preload_frames()
