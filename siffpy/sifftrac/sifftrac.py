import pandas as pd
import numpy as np
import os
from ..siffpy import SiffReader
from .. import siffutils

class FictracLog():
    """
    A class for reading in FicTrac output files.

    Currently, these are implemented as a .csv, though 
    I expect that that will change some time in the future.
    So I will try to make this as compatible with that as possible.

    Attributes
    ----------

    dataframe (pd.Dataframe) :

        A dataframe produced by reading in and parsing the path provided
        to the FicTrac log

    Methods
    -------

    align_to_imaging_data(siffreader, color_channel) : 

        Appends a column to the dataframe attribute that reflects
        the closest imaging frame, in time, to the FicTrac frame collected.
        Defaults to the lowest color channel

    """
    def __init__(self, filepath : str = None):
        if not filepath:
            #TODO: File navigation dialog
            pass

        if os.path.splitext(filepath)[-1] == '.csv':
            # It's a csv file and parse it as such
            self.dataframe = pd.read_csv(filepath, ',')

        if not hasattr(self,'dataframe'):
            raise NotImplementedError(
                "Could not set up a dataframe with information given"
                )

    def align_to_imaging_data(self, siffreader : SiffReader, color_channel : int = 0, **kwargs):
        """
        Goes through the frames in the file opened by the siffreader
        and identifies which frames in the FicTrac data are most
        closely aligned to it in time. Appends these as a column
        to self.dataframe.

        Color channel specification is useful because then you can directly
        reference the frame number you want (instead of later having)
        to shift by the number of colors in the image
        """
        if not siffreader.opened:
            raise AssertionError(
                "Siffreader object has no open .siff or .tiff file."
            )

        if not hasattr(self, 'dataframe'):
            raise Exception("No pandas Fictrac dataframe assigned.")

        if 'uses_seconds' in kwargs:
            # Undocumented kwarg for data from before the epoch value used seconds
            if kwargs['uses_seconds']:
                im_stamps_epoch = np.array(
                    siffreader.get_time(
                        frames = siffutils.framelist_by_color(siffreader.im_params, color_channel), 
                        reference = "epoch"
                    )
                )*1e9 # list of timestamps
        else:
            im_stamps_epoch = np.array(
                siffreader.get_time(
                    frames = siffutils.framelist_by_color(siffreader.im_params, color_channel), 
                    reference = "epoch"
                )
            )
        
        self.dataframe['aligned_imaging_frame'] = \
            pd.Series([
                np.searchsorted(im_stamps_epoch, fic_time)
                for fic_time in self.dataframe['timestamp']
                ]
            )
        