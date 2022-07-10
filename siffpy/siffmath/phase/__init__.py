import inspect, textwrap

from .phase_analyses import estimate_phase, fit_offset
from . import phase_estimates

def phase_alignment_functions(print_docstrings : bool = True)->None:
    """
    Prints the available methods for aligning a vector time series to a phase,
    as well as returning the string
    """
    print_string = f""
    memberfcns = inspect.getmembers(phase_estimates, inspect.isfunction)

    for member_fcn_info in memberfcns:
        fcn_name = member_fcn_info[0]
        fcn_call = member_fcn_info[1]
        print_string += f"\033[1m{fcn_name}\033[0m\n\n"
        print_string += f"\t{fcn_name}{inspect.signature(fcn_call)}\n\n"
        print_string += textwrap.indent(str(inspect.getdoc(fcn_call)),"\t\t")
        print_string += "\n\n"
    
    if print_docstrings:
        print(print_string)
    else:
        return memberfcns