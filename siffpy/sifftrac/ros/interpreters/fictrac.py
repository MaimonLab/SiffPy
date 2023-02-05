from pathlib import Path
from typing import Union

import polars as pl
import numpy as np

from siffpy.sifftrac.utils import BallParams
from siffpy.sifftrac.ros.ros_interpreter import ROSInterpreter, ROSLog
from siffpy.sifftrac.ros.config_file_params import ConfigParams, ConfigFileParamsMixin
from siffpy.sifftrac.ros.git_validation import GitConfig, GitValidatedMixin

FICTRAC_COLUMNS = [
    'timestamp',
    'frame_id',
    'frame_counter',
    'delta_rotation_cam_0',
    'delta_rotation_cam_1',
    'delta_rotation_cam_2',
    'delta_rotation_error',
    'delta_rotation_lab_0',
    'delta_rotation_lab_1',
    'delta_rotation_lab_2',
    'absolute_rotation_cam_0',
    'absolute_rotation_cam_1',
    'absolute_rotation_cam_2',
    'absolute_rotation_lab_0',
    'absolute_rotation_lab_1',
    'absolute_rotation_lab_2',
    'integrated_position_lab_0',
    'integrated_position_lab_1',
    'integrated_heading_lab',
    'animal_movement_direction_lab',
    'animal_movement_speed',
    'integrated_motion_0',
    'integrated_motion_1',
    'sequence_counter'
]

class FicTracLog(ROSLog):

    @classmethod
    def isvalid(cls, path : Path)->bool:
        """ Checks extension and column titles """
        valid = path.suffix == '.csv'
        cols = pl.scan_csv(path, sep=',', n_rows=1).columns
        valid *= all([col in cols for col in FICTRAC_COLUMNS])
        return valid

    def open(self, path : Path):
        if not self.isvalid(path):
            raise ValueError(f"""
                File {path} does not have the correct extension
                for {self.__class__.__name__} log files.
            """)
        
        self.df = pl.read_csv(path, sep=',')

class FicTracInterpreter(GitValidatedMixin, ConfigFileParamsMixin, ROSInterpreter):
    """ ROS interpreter for the ROSFicTrac node"""

    LOG_TYPE = FicTracLog
    LOG_TAG = '.csv'

    git_config = GitConfig(
        branch = 'main',
        commit_time = '2022-08-19 16:51:19-04:00',
        package = 'fictrac_ros2',
        repo_name = 'fictrac_ros2',
        executable = 'trackmovements'
    )

    config_params = ConfigParams(
        packages = ['fictrac_ros2'],
        executables={'fictrac_ros2' : ['trackmovements']},
    )

    def __init__(
            self,
            file_path : Union[str, Path],
            ball_params : BallParams = BallParams(),
        ):
        self.ball_params = ball_params
        # can be done appropriately
        super().__init__(file_path)

    @property
    def df(self)->Union[pl.DataFrame, pl.LazyFrame]:
        if hasattr(self.log, 'df'):
            return self.log.df

    @property
    def x_position(self)->np.ndarray:
        if self.experiment_config is None:
            return self.log.df['integrated_position_lab_0'].to_numpy()
        
    @property
    def y_position(self)->np.ndarray:
        if self.experiment_config is None:
            return self.log.df['integrated_position_lab_1'].to_numpy()
        
    @property
    def heading(self)->np.ndarray:
        if self.experiment_config is None:
            return self.log.df['integrated_heading_lab'].to_numpy()
        
    @property
    def timestamp(self)->np.ndarray:
        if self.experiment_config is None:
            return self.log.df['timestamp'].to_numpy()
    
    @property
    def movement_speed(self)->np.ndarray:
        if self.experiment_config is None:
            return self.log.df['animal_movement_speed'].to_numpy()
        