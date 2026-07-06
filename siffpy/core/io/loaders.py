# Functions for loading different types of files.
from typing import Union, Tuple, Optional, overload, Any
import pickle
import pathlib
import logging

from siffpy.core.utils.types import PathLike
from siffpy.core.flim import FLIMParams
from siffpy.core.utils.registration_tools import (
    RegistrationInfo, to_registration_info, to_reg_info_class,
    MROIRegistrationInfo, RegistrationInfoCollection, to_registration_info_collection,
)

@overload
def load_registration(
        siffio,
        im_params,
        filename : PathLike,
        mroi : None,
    )->RegistrationInfo:
    ...

@overload
def load_registration(
        siffio,
        im_params,
        filename : PathLike,
        mroi : Any,
    )->RegistrationInfoCollection:
    ...    

def load_registration(
        siffio,
        im_params,
        filename : Union[pathlib.Path, str],
        mroi : Optional[Any] = None,
    )->Optional[Union[RegistrationInfo, RegistrationInfoCollection]]:
    """
    Checks for a `RegistrationInfo` file associated with the
    passed `filename` by looking for any file with the suffix
    `_registration_info` in the same directory as the passed `filename`
    or its subdirectories. If such a file is found, it is loaded and returned as a
    `RegistrationInfo` object.

    ## Arguments

    - `siffio`: The `SiffIO` object to use for loading the registration information.
    - `im_params`: The `ImParams` object to use for loading the registration information.
    - `filename`: The filename to look for associated registration information.
    - `mroi`: Whether to look for MROI registration information (default: False).

    ## Returns

    - A `RegistrationInfo` object if registration information is found, or a `RegistrationInfoCollection`
        of `MROIRegistrationInfo`
        objects if `mroi` is True.
    - `None` if no registration information is found.
    """
    path = pathlib.Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    if (
        regpath := (
                    path.with_suffix("")/(path.stem+"_registration_info")
                ).with_suffix(
                    RegistrationInfo.REGISTRATION_INFO_SUFFIX
        )
    ).exists():
        if mroi is None:
            return to_registration_info(regpath, siffio, im_params)
    
    rdicts_iterator = path.with_suffix("").glob(path.stem + '_registration_info*')
    if mroi is not None and (next(rdicts_iterator, None) is not None):
        return to_registration_info_collection(path, siffio, im_params)
    
    if (regpath := path.with_suffix(".dict")).exists():
        reg_dict, ref_frames = load_registration_legacy(filename)
        ret_val : RegistrationInfo = to_reg_info_class('siffpy')(siffio, im_params)
        ret_val.yx_shifts = reg_dict
        ret_val.reference_frames = ref_frames
        ret_val.registration_color_channel = 0
        ret_val.save()
        return ret_val

    return None    

def load_registration_legacy(filename : str)->tuple:
    """
    Loads a registration dictionary and referrence frames from a file
    with the same name as the input file, but with a .dict extension.
    """
    path = pathlib.Path(filename)
    ret = []
    if (dictpath := path.with_suffix(".dict")).exists():
        with open(str(dictpath), 'rb') as dict_file:
            reg_dict = pickle.load(dict_file)
        if isinstance(reg_dict, dict):
        #    print("\n\n\tFound a registration dictionary for this image and importing it.\n")
            ret.append(reg_dict)
        else:
            logging.warning("\n\n\tPutative registration dict for this file is not of type dict.\n")
            ret.append(None)
    else:
        ret.append(None)
    if (refpath := path.with_suffix(".ref")).exists():
        with open(str(refpath), 'rb') as images_list:
            ref_ims = pickle.load(images_list)
        if isinstance(ref_ims, list):
        #    print("\n\n\tFound a reference image list for this file and importing it.\n")
            ret.append(ref_ims)
        else:
            logging.warning("\n\n\tPutative reference images object for this file is not of type list.\n", stacklevel=2)
            ret.append(None)
    else:
        ret.append(None)

    return tuple(ret)

def load_flim_params(
        filename : PathLike,
        glob_pattern : str = "*.flimparams",
    )->Tuple['FLIMParams']:
    """
    Returns a tuple of FLIMParams if any are stored in a directory
    with the name of the file, but with a .flimparams extension.

    Can use an alternate `glob_pattern` if preferred.
    """
    filename = pathlib.Path(filename)
    return tuple(
        FLIMParams.load(x)
        for x in filename.with_suffix("").glob(glob_pattern)
    )