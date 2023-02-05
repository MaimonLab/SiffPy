"""
The ROS interpreter object is a class for interpreting the various log
files for each type of ROS node. The idea is:

Each type of node logs data its own way, usually with characteristic
log file names, types, locations, etc. 
"""

from typing import Union, Type
from pathlib import Path
from abc import ABC, abstractmethod, abstractclassmethod

class ROSLog(ABC):

    def __init__(self, path : Path):
        self.path : Path = path
        self.open(path)

    @abstractclassmethod
    def isvalid(cls, path : Path)->bool:
        """ Returns whether a path points to a valid log file."""
        pass

    @abstractmethod
    def open(self, path : Path):
        pass

class ROSInterpreter(ABC):
    """
    The ROSInterpreter class is a class for interpreting the various log
    files for each type of ROS node.
    """
    LOG_TYPE : Type[ROSLog]
    LOG_TAG : str # file suffix

    def __init__(self, file_path : Union[str, Path]):
        """
        The constructor for the ROSInterpreter class.

        Parameters
        ----------
        file : str
            The path to the log file to be interpreted.
        """
        #self.file_path = file_path
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not (
                (self.__class__.LOG_TAG is None) or
                (file_path.suffix == self.__class__.LOG_TAG)
            ):
            raise ValueError(f"""
                File {file_path} does not have the correct extension
                for {self.__class__.__name__} log files.
            """)
        if not (file_path.exists()):
            raise ValueError(
                f"""File {file_path} does not exist."""
            )
        
        self.file_path : Path = file_path
        self.log = self.open(self.file_path)

    @classmethod
    def open(cls, file_path : Path)->ROSLog:
        """
        The open method returns a ROSLog object,
        which confirms that the file is of the correct type
        and provides access to the relevant attributes
        """
        return cls.LOG_TYPE(file_path)
    
    @classmethod
    def isvalid(cls, file_path : Path)->bool:
        """
        The isvalid method returns whether a file is of the correct type.
        """
        return cls.LOG_TYPE.isvalid(file_path)

        