# Functions for loading different types of files.

# These functions shouldn't be here for long, since
# soon they will be replaced by the new RegistrationInfo
# class.
import pickle
import pathlib
import logging

def load_registration(filename : str)->tuple:
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
    