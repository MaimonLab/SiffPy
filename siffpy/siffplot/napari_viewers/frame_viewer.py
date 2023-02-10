import logging
from typing import Callable, Iterable

import numpy as np
from qtpy import QtWidgets

from siffpy.core import SiffReader
from siffpy.siffplot.napari_viewers.napari_interface import NapariInterface

DASK = False
try:
    from dask import delayed
    import dask.array as da
    DASK = True
except ImportError:
    pass


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

        A SiffReader object linked to an open
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

    IMAGE_LAYER_NAME = 'Raw images (one FOV)'
    WINDOW_TITLE     = 'Frame Viewer'

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

        super().__init__(siffreader, *args, **kwargs, title = self.__class__.WINDOW_TITLE)
        # dumb bug prevents setting this in the init
        self.viewer.dims.axis_labels = siffreader.im_params.axis_labels

        if load_frames or (not DASK):
            stack = self._default_preload_frames()
        
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
                batch_iter = self._default_batch_iter()
            
            # function to get with each output of batch_iter
            if batch_fcn is None:
                batch_fcn = self._default_volume_get()
            
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

        self.viewer.add_image(
            data=stack,
            name=self.__class__.IMAGE_LAYER_NAME,
            scale = siffreader.im_params.scale,
            multiscale=False,
            channel_axis=channel_axis,
            contrast_limits = contrast,
            rgb = False
        )

    def _default_volume_get(self)->Callable:
        """
        Returns the default function for the batch function of the get_frames layer.

        To be implemented differently by other subclasses.
        """
        def volume_get(volume_idx):
            # Local function executed over and over
            return np.array(
                self.siffreader.get_frames(
                    frames=self.siffreader.im_params.flatten_by_timepoints(
                        timepoint_start = volume_idx,
                        timepoint_end = volume_idx+1,
                    ),
                    registration_dict=self.siffreader.registration_dict
                )
            ).reshape(self.siffreader.im_params.volume)

        return volume_get

    def _default_batch_iter(self)->Iterable:
        return range(self.siffreader.im_params.num_volumes)

    def _default_preload_frames(self)->np.ndarray:
        """
        Returns all frames loaded at the initiation of the reader.
        """
        logging.warn("Loading all frames. This might take a while...\n")
        
        frames = self.siffreader.get_frames(
            frames=self.siffreader.im_params.all_frames,
            registration_dict= self.siffreader.registration_dict
        )
        return np.array(
            frames[:self.siffreader.im_params.final_full_volume_frame]
        ).reshape((-1,*self.siffreader.im_params.stack))

        #self.show_roi_widget = _make_show_roi_widget()


        #self.viewer.window.add_dock_widget(self.show_roi_widget,name='Show ROIs')

# def _make_show_roi_widget()->QtWidgets.QHBoxLayout:
#     """ Returns the appropriately formatted QHBoxLayout object """
#     layout = QtWidgets.QHBoxLayout()
#     layout.addWidget(QtWidgets.QPushButton("Show ROIs"))
#     return layout