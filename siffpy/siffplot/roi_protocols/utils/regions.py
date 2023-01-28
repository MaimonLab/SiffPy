from enum import Enum
from typing import Callable
from dataclasses import dataclass
import inspect

class RegionEnum(Enum):
    ELLIPSOID_BODY = 'Ellipsoid body'
    FAN_SHAPED_BODY = 'Fan-shaped body'
    PROTOCEREBRAL_BRIDGE = 'Protocerebral bridge'
    NODULI = 'Noduli'
    GENERIC = 'Generic'

@dataclass
class SegmentationFunction():
    name : str
    func : Callable
    
@dataclass
class Region():
    alias_list : list[str]
    module : object
    default_fcn_str : str
    region_enum : RegionEnum

    @property
    def functions(self)->list[SegmentationFunction]:
        return list(
            SegmentationFunction(*func_tup) for func_tup in
            inspect.getmembers(self.module, inspect.isfunction)
        )

    @property
    def default_fcn(self) -> Callable:
        return next(fcn.func for fcn in self.functions if fcn.name == self.default_fcn_str)