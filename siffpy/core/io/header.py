import logging, re

from siffpy.core.utils import ImParams

def vector_to_list(vector, vec_num : int = 0, ret_type=float):
    """
    list = vector_to_list(vector, type=float)

    Interprets either a MATLAB column vector or row vector as a python list

    Inputs
    ------
    vector (string):
        String version of a MATLAB vector, e.g. '[1;2]' or '[1,5,5]'

    vec_num (int, optional):
        Which vector element to use.

    type (optional):
        type of numbers (int, float)

    Returns
    ------
    list (list):
        List version of the vector input
    """
    # if it's just a number, then we don't need to worry about this
    try:
        return ret_type(vector)
    except:
        pass
    
    betwixt_brackets = re.findall(r"^.*\[(.*)\].*$",vector)
    if not betwixt_brackets:
        return None
    if len(betwixt_brackets) > 1 and vec_num == 0:
        logging.warning("Ambiguous string. Using first matching vector.")
    col_split = betwixt_brackets[vec_num].split(';')
    row_split = betwixt_brackets[vec_num].split(' ')

    if (len(col_split) > 1) and (len(row_split) > 1):
        raise ValueError("Input string could not be parsed into a row vector or column vector.")
    if len(col_split)>1:
        return [ret_type(element) for element in col_split]
    else:
        return [ret_type(element) for element in row_split]

def matrix_to_listlist(matrix : str, vec_num : int = 0, ret_type = float) -> list[list]:
    """
    Converts the string representation of a MATLAB matrix into a list of lists
    """
    try:
        return ret_type(matrix)
    except:
        pass
    
    betwixt_brackets = re.findall(r"^.*\[(.*)\].*$",matrix)
    if not betwixt_brackets:
        return None
    if len(betwixt_brackets) > 1 and vec_num == 0:
        logging.warning("Ambiguous string. Using first matching vector.")
    col_split = betwixt_brackets[vec_num].split(';')
    row_split = betwixt_brackets[vec_num].split(' ')

    if (len(col_split) > 1) and (len(row_split) > 1):
        return [[ret_type(element) for element in column.split(" ")] for column in col_split]
    # if it's just a vector, use the vector parser
    else:
        return vector_to_list(matrix, ret_type)
    
def header_data_to_nvfd(hd):
    return {entry.split(" = ")[0] : (entry.split(" = ")[1] if (len(entry.split(" = "))>1) else None) for entry in hd["Non-varying frame data"].split("\n")}

def header_data_to_roi_string(hd : str) -> dict:
    """ Iterate through the many layers of the ROI strings to return the appropriate dict """
    return eval(hd['ROI string'].replace("null", "None"))

def header_to_imparams(header : str, num_frames : int = None)->ImParams:
    """
    Returns an object containing the most important image parameters
    """
    header_dict = header_data_to_nvfd(header)
    ROI_group_data = header_data_to_roi_string(header)
    
    im_pars = ImParams.from_dict(header_dict, num_frames = num_frames)
    im_pars.add_roi_data(ROI_group_data)
    
    return im_pars