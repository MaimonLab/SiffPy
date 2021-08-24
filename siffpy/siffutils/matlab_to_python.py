"""
Functionality made to convert MATLAB output
in string form into Python types for analysis.
I'll add to this as it becomes relevant

SCT March 27, 2021
"""

import re
import warnings

def vector_to_list(vector, ret_type=float):
    """
    list = vector_to_list(vector, type=float)

    Interprets either a MATLAB column vector or row vector as a python list

    Inputs
    ------
    vector (string):
        String version of a MATLAB vector, e.g. '[1;2]' or '[1,5,5]'

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
    if len(betwixt_brackets) > 1:
        warnings.warn("Ambiguous string. Using first matching vector.")
    col_split = betwixt_brackets[0].split(';')
    row_split = betwixt_brackets[0].split(' ')

    if (len(col_split) > 1) and (len(row_split) > 1):
        raise ValueError("Input string could not be parsed into a row vector or column vector.")
    if len(col_split)>1:
        return [ret_type(element) for element in col_split]
    else:
        return [ret_type(element) for element in row_split]

def matrix_to_listlist(matrix, ret_type = float) -> list[list]:
    try:
        return ret_type(matrix)
    except:
        pass
    
    betwixt_brackets = re.findall(r"^.*\[(.*)\].*$",matrix)
    if not betwixt_brackets:
        return None
    if len(betwixt_brackets) > 1:
        warnings.warn("Ambiguous string. Using first matching vector.")
    col_split = betwixt_brackets[0].split(';')
    row_split = betwixt_brackets[0].split(' ')

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


    