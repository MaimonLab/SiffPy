from pathlib import Path

from siffreadermodule import SiffIO
from siffpy.core.utils import ImParams
from siffpy.core.utils.types import PathLike
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationType, RegistrationInfo
)

def to_reg_info_class(
    stringname : str
)->type[RegistrationInfo]:
    """ Returns a class of registration info """
    registration_type = RegistrationType(stringname)

    cls = None
    if registration_type == RegistrationType.Caiman:
        try:
            from siffpy.core.utils.registration_tools.caiman import CaimanRegistrationInfo
            cls = CaimanRegistrationInfo
        except ImportError as e:
            raise ImportError(
                f"""Failed to import caiman registration info.
                Likely need to install caiman. Error:
                {e.with_traceback(e.__traceback__)}
                """
            )
    elif registration_type == RegistrationType.Suite2p:
        try:
            from siffpy.core.utils.registration_tools.suite2p import Suite2pRegistrationInfo
            cls = Suite2pRegistrationInfo
        except ImportError as e:
            raise ImportError(
                f"""Failed to import Suite2p registration info.
                Likely need to install suite2p. Error:
                {e.with_traceback(e.__traceback__)}
                """
            )
    elif registration_type == RegistrationType.Siffpy:
        from siffpy.core.utils.registration_tools.siffpy import SiffpyRegistrationInfo
        cls = SiffpyRegistrationInfo
    elif registration_type == RegistrationType.Average:
        raise NotImplementedError("Haven't implemented average registration yet.")
    elif registration_type == RegistrationType.Other:
        raise ValueError("""
            This function cannot return a RegistrationInfo
            of type `Other`. Please define and instatiate the
            RegistrationInfo object yourself using the
            CustomRegistrationInfo class (accesible with
            `from siffpy.core.utils.registration_tools.registration_info import CustomRegistrationInfo`)
            """
        )
    else:
        raise ValueError(f"Unrecognized registration type: {registration_type}")
    return cls

def to_registration_info(
        path : PathLike, 
        siffio : SiffIO = None,
        im_params : ImParams = None,
    )->RegistrationInfo:
    """
    Returns a registration info object from a path
    """
    if isinstance(path, str):
        path = Path(path)

    as_dict = RegistrationInfo.load_as_dict(path)
    registration_type = as_dict['registration_type']

    cls = to_reg_info_class(registration_type)
    
    reginfo = cls(
        siffio = siffio,
        im_params = im_params,
    )

    reginfo.from_dict(as_dict)
    return reginfo
    
