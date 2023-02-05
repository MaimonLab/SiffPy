"""
A class which parses a configuration file, discerns
which types of interpreters are needed, instantiates them,
and lets them find their appropriate logs
"""
from pathlib import Path
from typing import Union

class Experiment():

    def __init__(self, path : Union[str, Path]):
        raise NotImplementedError("Not implemented yet.")