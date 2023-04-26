import pandas as pd
import numpy as np
import os, logging

import ruamel.yaml

from siffpy.core import SiffReader
from siffpy.core.timetools import SEC_TO_NANO, NANO_TO_SEC
from siffpy.core.utils import circle_fcns
from siffpy.sifftrac.utils import BallParams


_ORIGINAL_FICTRAC_ROS_ZERO_HEADING = 3.053108549228689 # an error in the original projector_driver ROS2 code resulted in an
# incorrect map between fictrac heading and the bar position relative to the fly.


class FictracLog():
    """
    A class for reading in FicTrac output files.

    Currently, these are implemented as a .csv, though 
    I expect that that will change some time in the future.
    So I will try to make this as compatible with that as possible.

    Attributes
    ----------

    dataframe (pd.DataFrame) :

        A dataframe produced by reading in and parsing the path provided
        to the FicTrac log, and may interact with the siffreader to add
        columns + analyses. Most operations on this dataframe are done
        in place, so be careful to use deep copies if you want to play
        around with data outside of functionality implemented by this class.

    siffreader (siffpy.SiffReader) : 

        A linked siffreader with an open .siff file

    filename (str) :

        The path to the file with the data for the trajectory represented here
        (may be useful for dynamic reading in the future)

    Methods
    -------

    align_to_image_time(siffreader) : 

        Appends a column to the dataframe that converts epoch time to
        experiment time, as reflected by the frame timestamps in the siff
        or tiff file.

    align_to_imaging_data(siffreader, color_channel) : 

        Appends a column to the dataframe attribute that reflects
        the closest imaging frame, in time, to the FicTrac frame collected.
        Defaults to the lowest color channel

    """
    def __init__(self, filepath : str = None, ballparams : BallParams = None, suppress_warnings : bool = False):
        if not filepath:
            #TODO: File navigation dialog
            pass
        
        if ballparams is None:
            ballparams = BallParams()

        self.__find_projector_specification(filepath)

        self.filename = filepath
        self.ballparams = ballparams

        if os.path.splitext(filepath)[-1] == '.csv':
            # It's a csv file and parse it as such
            if not suppress_warnings:
                logging.warning("Converting dataframe coordinates to mm and rotating so that the bar is in the +y direction.")
            self._dataframe = pd.read_csv(filepath, sep=',')
            radius = self.ballparams.radius
            self._dataframe['integrated_position_lab_0']*=radius
            self._dataframe['integrated_position_lab_1']*=radius
            self._dataframe['animal_movement_speed'] *= radius

            df_copy = self.get_dataframe_copy()

            self._dataframe['integrated_position_lab_0'] = -1.0*df_copy['integrated_position_lab_1']
            self._dataframe['integrated_position_lab_1'] = df_copy['integrated_position_lab_0']

            if self._OLD_PROJECTOR_DRIVER:
                self._dataframe['integrated_position_lab_0'], self._dataframe['integrated_position_lab_1'] = (
                    (
                        np.cos(_ORIGINAL_FICTRAC_ROS_ZERO_HEADING) * self._dataframe['integrated_position_lab_0'] +
                        np.sin(_ORIGINAL_FICTRAC_ROS_ZERO_HEADING) * self._dataframe['integrated_position_lab_1']
                    ),
                    (
                        -1.0*np.sin(_ORIGINAL_FICTRAC_ROS_ZERO_HEADING) * self._dataframe['integrated_position_lab_0'] +
                        np.cos(_ORIGINAL_FICTRAC_ROS_ZERO_HEADING) * self._dataframe['integrated_position_lab_1']
                    )
                )
                
                pass

        if not hasattr(self,'_dataframe'):
            raise NotImplementedError(
                "Could not set up a dataframe with information given"
            )

    def align_to_imaging_data(self, siffreader : SiffReader = None, color_channel : int = 0, **kwargs):
        """
        Goes through the frames in the file opened by the siffreader
        and identifies which frames in the FicTrac data are most
        closely aligned to it in time. Appends these as a column
        to self._dataframe.

        Color channel specification is useful because then you can directly
        reference the frame number you want (instead of later having)
        to shift by the number of colors in the image
        """
        if (siffreader is None):
            if hasattr(self, 'siffreader'):
                siffreader = self.siffreader
            else:
                raise AssertionError(
                    "No SiffReader object linked"
                )

        if not siffreader.opened:
            if hasattr(self, 'siffreader'):
                if not self.siffreader.opened:
                    raise AssertionError("Siffreader has no open file")
            else:
                raise AssertionError("Siffreader has no open file")

        if not hasattr(self, 'siffreader'):
            self.siffreader = siffreader

        if not hasattr(self, '_dataframe'):
            raise Exception("No pandas Fictrac dataframe assigned.")

        # Only need one color channel, since color frames have shared tiempoints
        framelist = siffreader.im_params.framelist_by_color(color_channel) 

        if 'uses_seconds' in kwargs:
            # Undocumented kwarg for data from before the epoch value used nanoseconds
            if kwargs['uses_seconds']:
                im_stamps_epoch = np.array(
                    siffreader.get_time(
                        frames = framelist, 
                        reference = "epoch"
                    ) * SEC_TO_NANO 
                ) # list of timestamps
        else:
            im_stamps_epoch = np.array(
                siffreader.get_time(
                    frames = framelist, 
                    reference = "epoch"
                )
            )

        aligned_frame = pd.Series([
                framelist[np.searchsorted(im_stamps_epoch, fic_time)-1]
                if (0 < np.searchsorted(im_stamps_epoch, fic_time) < len(im_stamps_epoch)) # nan before and after imaging experiment
                else np.nan
                for fic_time in self._dataframe['timestamp']
                ],
                name='aligned_imaging_frame'
            )

        # this gets me every goddamn time
        aligned_frame.index = self._dataframe.index
        self._dataframe['aligned_imaging_frame'] = aligned_frame
        
    def align_to_image_time(self, siffreader : SiffReader = None, **kwargs)->None:
        """
        Adds a new column to the dataframe that is aligned to
        the imaging experiment's "experiment time", which
        1) is in seconds
        2) is 0'd at the start of acquisition
        """

        if (siffreader is None):
            if hasattr(self, 'siffreader'):
                siffreader = self.siffreader
            else:
                raise AssertionError(
                    "No SiffReader object linked"
                )

        if not siffreader.opened:
            if hasattr(self, 'siffreader'):
                if not self.siffreader.opened:
                    raise AssertionError("Siffreader has no open file")
            else:
                raise AssertionError("Siffreader has no open file")

        if not hasattr(self, 'siffreader'):
            self.siffreader = siffreader

        if 'uses_seconds' in kwargs:
            # Undocumented kwarg for data from before the epoch value used seconds
            if kwargs['uses_seconds']:
                first_frame_epoch = np.array(
                    siffreader.get_time(
                        frames = [0], 
                        reference = "epoch"
                    )
                )*SEC_TO_NANO # list of timestamps
        else:
            first_frame_epoch = np.array(
                siffreader.get_time(
                    frames = [0],
                    reference = "epoch"
                )
            )
        
        first_frame_expt = siffreader.get_time(frames = [0], reference = "experiment")

        offset = first_frame_expt*SEC_TO_NANO - first_frame_epoch # Time between first frame timestamp in epoch and the experiment onset

        self._dataframe['image_time'] = (self._dataframe['timestamp'] + offset)/SEC_TO_NANO
        
    def discard_pre_and_post_imaging_data(self) -> None:
        """
        To shrink the dataframe --- may not often be necessary
        """
        if not hasattr(self,'siffreader'):
            raise RuntimeError("FictracLog not linked to an image file")

        if not 'image_time' in self._dataframe.columns:
            print("Aligning to image time")
            self.align_to_image_time(self.siffreader)

        self._dataframe = self._dataframe.loc[self._dataframe['image_time'] >= 0]

        if 'aligned_imaging_frame' in self._dataframe.columns:
            self._dataframe.dropna(axis='index', inplace=True, subset=['aligned_imaging_frame'])

            #why isn't there an inplace version of this...
            self._dataframe = self._dataframe.astype({'aligned_imaging_frame':'uint64'}, copy=False)
            
        self._dataframe.reset_index(drop=True, inplace=True)

    def downsample_to_imaging(self, method : str = 'downsample', groupby : str = 'volume')->pd.DataFrame:
        """
        RETURNS (does not store!!) a downsampled dataframe according to the method provided.

        Arguments
        --------

        method : str

            How to perform the downsampling. Available options:

                - 'downsample'          : takes the point nearest in time to each image frame
                - 'average' or 'mean'   : takes the average of all points between the correspond image frame and the next

        groupby : str
            
            Whether to return a sample for every volume or for every individual frame. Options:

                - 'volume' : one row for every volume
                - 'frame'  : one row for every frame

        """
        if not hasattr(self,'siffreader'):
            raise RuntimeError("FictracLog not linked to an image file")

        if not 'image_time' in self._dataframe.columns:
            print("Aligning to image time")
            self.align_to_image_time(self.siffreader)

        if not 'aligned_imaging_frame' in self._dataframe.columns:
            self.align_to_imaging_data(self.siffreader)

        if not groupby in ['volume', 'frame']:
            raise ValueError(f"Invalid groupby {groupby}. Must be either 'volume' or 'frame'.")

        if method == 'mean':
            method = 'average'

        if not method in ['average','downsample']:
            raise ValueError(f"Invalid method of downsampling data '{method}'. Consult help(sifftrac.log_interpreter.fictraclog.FictracLog.downsample_to_imaging)")

        
        copied_df = self.get_dataframe_copy()

        new_frame_indices = np.insert(np.where(np.diff(copied_df['aligned_imaging_frame']))[0]+1,0,0) # first index of each new imaging frame

        if groupby == 'frame':
            if method == 'downsample':
                return copied_df.iloc[new_frame_indices]
            
            if method == 'average':
                return copied_df.groupby('aligned_imaging_frame').mean()

        if groupby == 'volume':
            copied_df['aligned_volume'] = copied_df['aligned_imaging_frame']//self.siffreader.im_params.frames_per_volume
            if method == 'downsample':
                new_volume_indices = np.insert(np.where(np.diff(copied_df['aligned_volume']))[0]+1,0,0) # first index of each new volume
                return copied_df.iloc[new_volume_indices]
            if method == 'average':
                return copied_df.groupby('aligned_volume').mean()
        
        raise RuntimeError("Unanticipated error in source code argument checking, slipped through the cracks.")
    
    def unwrap_heading(self)->np.ndarray:
        """
        Unwraps the heading in the internal dataframe, making a new column ('unwrapped_heading'), but also returns
        a reference to its values
        """

        hd = self._dataframe['integrated_heading_lab']

        unwrapped = circle_fcns.circ_unwrap(hd)

        self._dataframe['unwrapped_heading'] = unwrapped
        return unwrapped

    def __getattr__(self, name: str) -> object:
        """
        For safe copying of dataframes, FORCES you to copy.
        """
        if name == 'dataframe':
            logging.warning("""

            ONLY DEEP COPIES OF THE FICTRAC DATAFRAME ARE RETURNED FOR EXTERNAL USE.\n\n
            THIS IS BECAUSE THE FICTRACLOG CLASS PERFORMS MOST OPERATIONS INPLACE
            SO THE RETURNED DATAFRAME WOULD OTHERWISE BE VERY LIKELY TO CHANGE
            AND I DON'T WANT THAT TO BE A SURPRISE!
            \n\nIF YOU WANT A COPY THAT CONTINUES TO POINT
            TO THE SAME DATA USE THE GET_DATAFRAME_REFERENCE METHOD. OTHERWISE USE GET_DATAFRAME_COPY
            TO MAKE THIS EXPLICIT.
            
            """
            )
            return self._dataframe.copy(deep=True)
        else:
            return super().__getattribute__(name)

    def get_dataframe_reference(self) -> pd.DataFrame:
        """
        Returns a reference to the internal dataframe to allow
        in-place modifications.
        """
        return self._dataframe

    def get_dataframe_copy(self) -> pd.DataFrame:
        """
        Returns a copy of the internal dataframe.
        
        Overrode the __getattr__ and made this an explicit method
        to avoid suprises when this class adjusts the dataframe in place.

        To get a reference to the dataframe, use get_dataframe_reference.
        """
        if hasattr(self,'_dataframe'):
            return self._dataframe.copy(deep=True)
        else:
            raise AttributeError("Dataframe attribute does not exist.")

    def __hash__(self) -> int:
        return hash(self.filename)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FictracLog):
            return False
        return self.__hash__() == other.__hash__()

    def __find_projector_specification(self, filepath : str)->None:
        """
        Needed for backwards compatibility. If there's no projector specification
        file, this is from an earlier era where there was a mistake in the heading-
        to-image map
        """

        self._OLD_PROJECTOR_DRIVER = True # Default behavior presumes the old spec

        filedir = os.path.split(filepath)[0]
        potential_specs = [fname for fname in os.listdir(filedir) if fname.endswith("_specifications.yaml")]
        if len(potential_specs) < 1:
            return
        if len(potential_specs) > 1:
            logging.warning(f"More than one potential spec file identified! Using: {potential_specs[0]}")
        
        ruamel_yaml = ruamel.yaml.YAML(typ="safe")
        ruamel_yaml.default_flow_style = False
        
        presumed_spec = os.path.join(filedir,potential_specs[0])
        with open(presumed_spec) as f:
            spec_dict = ruamel_yaml.load(f)

        if spec_dict['start_bar_in_front']:
            self._OLD_PROJECTOR_DRIVER = False


