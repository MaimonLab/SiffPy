# Functions for loading different types of files.
from typing import Union
import pickle
import pathlib
import logging

from siffpy.core.utils.registration_tools import (
    RegistrationInfo, to_registration_info, to_reg_info_class
)

def load_registration(
        siffio,
        im_params,
        filename : Union[pathlib.Path, str]
    )->RegistrationInfo:
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
        return to_registration_info(regpath, siffio, im_params)
    if (regpath := path.with_suffix(".dict")).exists():
        reg_dict, ref_frames = load_registration_legacy(filename)
        ret_val = to_reg_info_class('siffpy')(siffio, im_params)
        ret_val.yx_shifts = reg_dict
        ret_val.reference_frames = ref_frames
        return ret_val

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
    