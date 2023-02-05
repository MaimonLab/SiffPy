from pathlib import Path
from typing import Union

import polars as pl
import numpy as np
from ruamel.yaml import YAML

from siffpy.sifftrac.utils import BallParams
from siffpy.sifftrac.ros.ros_interpreter import ROSInterpreter, ROSLog
from siffpy.sifftrac.ros.git_validation import GitConfig, GitValidatedMixin
from siffpy.sifftrac.ros.config_file_params import ConfigParams, ConfigFileParamsMixin

class ProjectorLog(ROSLog):

    @classmethod
    def isvalid(cls, path : Path)->bool:
        """ Checks extension, ideally do more """
        valid = path.suffix == '.yaml'
        #y = YAML()
        #y.load(path)

        return valid

    def open(self, path : Path):
        if not self.isvalid(path):
            self.OLD_PROJECTOR_SPEC = True
            return
    
        y = YAML()
        info = y.load(path)
        self.OLD_PROJECTOR_SPEC = not (info['start_bar_in_front'])

class ProjectorInterpreter(GitValidatedMixin, ROSInterpreter):
    """ ROS interpreter for the ROSFicTrac node"""

    LOG_TAG = '.yaml'
    LOG_TYPE = ProjectorLog

    git_config = [
        GitConfig(
            branch = 'set_parameters_executable',
            commit_time = '2023-01-06 14:28:51-05:00',
            package = 'projector_driver',
            repo_name = 'projector_driver',
            executable = 'projector_bar',
        ),
        GitConfig(
            branch = 'set_parameters_executable',
            commit_time = '2023-01-06 14:28:51-05:00',
            package = 'dlpc_projector_settings',
            repo_name = 'projector_driver',
            executable = 'dlpc_projector_settings',
        ),
    ]

    def __init__(
            self,
            file_path : Union[str, Path],
        ):
        self.exp_params = None #TODO implement this so that coordinate transforms
        # can be done appropriately
        super().__init__(file_path)

    @property
    def OLD_PROJECTOR_SPEC(self)->bool:
        if hasattr(self.log, 'OLD_PROJECTOR_SPEC'):
            return self.log.OLD_PROJECTOR_SPEC
        else:
            return True