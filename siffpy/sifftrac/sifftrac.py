import pandas as pd
import numpy as np
import os
from ..siffpy import SiffReader

class FictracLog():
    """
    A class for reading in FicTrac output files.

    Currently, these are implemented as a .csv, though 
    I expect that that will change some time in the future.
    So I will try to make this as compatible with that as possible.

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

    def align_to_imaging_data(self, siffreader : SiffReader):
        """
        Goes through the frames in the file opened by the siffreader
        and identifies which frames in the FicTrac data are most
        closely aligned to it in time. Appends these as a column
        to self.dataframe
        """
        if not siffreader.opened:
            raise AssertionError(
                "Siffreader object has no open .siff or .tiff file."
            )

        t_stamps = siffreader.get_time(reference = "epoch") # list of timestamps