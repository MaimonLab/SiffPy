from pathlib import Path

from typing import TYPE_CHECKING, Iterable

#from siffreadermodule import SiffIO
from siffpy.core.utils import ImParams
from siffpy.core.utils.types import PathLike
from siffpy.core.utils.registration_tools.registration_info import (
    RegistrationType, RegistrationInfo, MROIRegistrationInfo,
)

if TYPE_CHECKING:
    from corrosiffpy import SiffIO


class RegistrationInfoCollection(dict[str, MROIRegistrationInfo]):
    """
    A collection of `MROIRegistrationInfo` objects, indexed by their `roiUuid`.
    """
    def __init__(self, mroi_list : Iterable[MROIRegistrationInfo]):
        super().__init__()
        for mroi in mroi_list:
            self[mroi.roiUuid] = mroi

    def __getitem__(self, idx : str) -> MROIRegistrationInfo:
        return super().__getitem__(idx)

def to_reg_info_class(
    stringname : str,
    mroi : bool = False,
)->'type[RegistrationInfo]':
    """
    Returns a class of registration info.

    If `mroi` is False, returns the standard registration info class corresponding to `stringname`.
    If `mroi` is True, returns the mROI registration info class corresponding to
    `stringname`, if it exists.
    If no mROI equivalent class exists for the registration type specified
    by `stringname`, raises a `NotImplementedError`.
    
    """
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
    
    if mroi:
        if cls is not None:
            if cls.mroi_class() is not None:
                mroi_class : type[MROIRegistrationInfo] = cls.mroi_class() # type: ignore
                return mroi_class
            else:
                raise NotImplementedError(
                    f"Registration type {registration_type} does not have an mROI equivalent implemented yet."
                )
    return cls

def to_registration_info(
        path : PathLike, 
        siffio : 'SiffIO',
        im_params : ImParams,
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
    
def to_registration_info_collection(
    path : PathLike,
    siffio : 'SiffIO',
    im_params : ImParams,
    ) -> RegistrationInfoCollection:
    """
    Loads a collection of `MROIRegistrationInfo` objects from a file at `path`
    and returns them as a `RegistrationInfoCollection`.

    ## Arguments
    - `path`: The path to the directory containing the registration information objects.
    - `siffio`: The `SiffIO` object to use for loading the registration information.
    - `im_params`: The `ImParams` object to use for loading the registration information.

    ## Returns
    - A `RegistrationInfoCollection` containing the loaded `MROIRegistrationInfo` objects.
    Raises a `FileNotFoundError` if no registration information is found at the specified path
    """
    path = Path(path)
    rdicts_iterator = path.with_suffix("").glob(path.stem + '_registration_info*')
    mroi_reg_infos = []
    for rdict_path in rdicts_iterator:
        if rdict_path.suffix == RegistrationInfo.REGISTRATION_INFO_SUFFIX:
            as_dict = RegistrationInfo.load_as_dict(rdict_path)
            registration_type = as_dict['registration_type']

            cls = to_reg_info_class(registration_type, mroi=True)
            as_dict = cls.load_as_dict(rdict_path)

            im_roi = next(roi for roi in im_params.imaging_rois if roi.roiUuid == as_dict['roiUuid'])
            reginfo = cls(
                siffio = siffio,
                im_params = im_params,
                mroi = im_roi,
            )

            reginfo.from_dict(as_dict)
            mroi_reg_infos.append(reginfo)
    if len(mroi_reg_infos) == 0:
        raise FileNotFoundError(f"No registration infos found at {path}")
    return RegistrationInfoCollection(mroi_reg_infos)
    