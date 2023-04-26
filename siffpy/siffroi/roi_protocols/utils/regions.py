from enum import Enum
from typing import Callable, Union
from dataclasses import dataclass
import inspect

from siffpy.siffroi.roi_protocols.roi_protocol import ROIProtocol

class RegionEnum(Enum):
    ELLIPSOID_BODY = 'Ellipsoid body'
    FAN_SHAPED_BODY = 'Fan-shaped body'
    PROTOCEREBRAL_BRIDGE = 'Protocerebral bridge'
    NODULI = 'Noduli'
    GENERIC = 'Generic'
    
@dataclass
class Region():
    alias_list : list[str]
    module : object
    default_fcn_str : str
    region_enum : RegionEnum

    @property
    def protocols(self)->list[ROIProtocol]:
        return list(
            protocol[1]()
            for protocol in
            inspect.getmembers(self.module, inspect.isclass)
            if (
                issubclass(protocol[1], ROIProtocol) and 
                protocol[1] != ROIProtocol
            )
        )
    
    @property
    def default_protocol(self)->ROIProtocol:
        return next(
            protocol for protocol in self.protocols
            if protocol.name == self.default_fcn_str
        )
