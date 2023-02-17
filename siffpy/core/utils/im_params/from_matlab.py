import re
import logging

def vector_to_list(vector, vec_num : int = 0):
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
        return int(vector)
    except ValueError:
        try:
            return float(vector)
        except:
            pass
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
    ret_arr = []

    # This sucks and is ugly
    if len(col_split)>1:
        for element in col_split:
            try:
                val = int(element)
            except ValueError:
                val = float(element)
        ret_arr.append(val)
        return ret_arr
    else:
        for element in row_split:
            try:
                val = int(element)
            except ValueError:
                val = float(element)
        ret_arr.append(val)
        return ret_arr

def matrix_to_listlist(matrix : str, vec_num : int = 0) -> list[list]:
    """
    Converts the string representation of a MATLAB matrix into a list of lists
    """
    try:
        return int(matrix)
    except ValueError:
        try:
            return float(matrix)
        except:
            pass
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
        retlist = []
        for column in col_split:
            collist = []
            for element in column.split(" "):
                try:
                    val = int(element)
                except ValueError:
                    val = float(element)
                collist.append(val)
            retlist.append(collist)
        return retlist
    # if it's just a vector, use the vector parser
    else:
        return vector_to_list(matrix, vec_num)
    
def contains_vector(in_string : str)->bool:
    return re.match(r"^.*\[(.*)\].*$",in_string) is not None