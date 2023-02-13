from typing import Union
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationType
)

def to_registration_info(
        registration_type : Union[str,RegistrationType],
    )->type:
    """
    Returns a registration info object of the specified type.

    To view options, select any from the
    siffpy.core.utils.registration_tools.registration_info.RegistrationType
    enum.
    """
    if isinstance(registration_type, str):
        registration_type = RegistrationType(registration_type)

    if registration_type == RegistrationType.Caiman:
        try:
            from siffpy.core.utils.registration_tools.caiman import CaimanRegistrationInfo
        except ImportError as e:
            raise ImportError(
                f"""Failed to import caiman registration info.
                Likely need to install caiman. Error:
                {e.with_traceback(e.__traceback__)}
                """
            )
        return CaimanRegistrationInfo
    elif registration_type == RegistrationType.Suite2p:
        try:
            from siffpy.core.utils.registration_tools.suite2p import Suite2pRegistrationInfo
        except ImportError as e:
            raise ImportError(
                f"""Failed to import Suite2p registration info.
                Likely need to install suite2p. Error:
                {e.with_traceback(e.__traceback__)}
                """
            )
        return Suite2pRegistrationInfo
    elif registration_type == RegistrationType.Siffpy:
        from siffpy.core.utils.registration_tools.siffpy import SiffpyRegistrationInfo
        return SiffpyRegistrationInfo
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
