from pathlib import Path

import polars as pl

from siffpy.sifftrac.ros.ros_interpreter import ROSInterpreter, ROSLog

class FicTracLog(ROSLog):

    @classmethod
    def isvalid(cls, path : Path)->bool:
        r_val = True
        if path.suffix == '.csv':
            r_val *= True
        pl.read_csv(path.__str__())
        return r_val

    def open(self, path : Path):
        self.data = pl.read_csv(path, sep='\t')

class FicTracInterpreter(ROSInterpreter):
    pass