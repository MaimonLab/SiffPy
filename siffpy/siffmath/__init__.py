import numpy as np
import inspect, textwrap

from . import phase

def estimate_phase(vector_series : np.ndarray, *args, method='pva', error_estimate = False, **kwargs)->np.ndarray:
    """
    Takes a time series of vectors (dimensions x time) and, assuming they
    correspond to a circularized signal, returns a 'phase' estimate, according
    to the method selected.

    Arguments
    ---------

    vector_series : np.ndarray

        An array of shape (D,T) where T is time bins and D is discretized elements of a 1-dimensional ring.

    method : str

        Available methods:

            - pva : Population vector average

    """

    if error_estimate:
        raise NotImplementedError("Error estimate on the phase is not yet implemented.")

    try:
        if not callable(getattr(phase, method)): # check that the method IS callable
            raise ValueError(f"No phase estimate method {method} in SiffMath module {phase}")
    except AttributeError as e:
        raise ValueError(f"No phase estimate method {method} in SiffMath module {phase}." +
        "To see available methods, call siffmath.phase_alignment_functions()")

    phase_method = getattr(phase, method)
    return phase_method(vector_series, *args, error_estimate = error_estimate, **kwargs)

def phase_alignment_functions()->None:
    """
    Prints the available methods for aligning a vector time series to a phase,
    as well as returning the string
    """
    print_string = f""
    memberfcns = inspect.getmembers(phase, inspect.isfunction)

    for member_fcn_info in memberfcns:
        fcn_name = member_fcn_info[0]
        fcn_call = member_fcn_info[1]
        print_string += f"\033[1m{fcn_name}\033[0m\n\n"
        print_string += f"\t{fcn_name}{inspect.signature(fcn_call)}\n\n"
        print_string += textwrap.indent(str(inspect.getdoc(fcn_call)),"\t\t")
        print_string += "\n\n"
    
    print(print_string)