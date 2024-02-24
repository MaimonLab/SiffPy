from typing import List
import inspect
import textwrap

from siffpy.siffmath.phase import phase_alignment_functions # noqa: F401
from siffpy.siffmath.phase.phase_estimates import estimate_phase # noqa: F401
from siffpy.siffmath.flim import FlimTrace # noqa: F401
from siffpy.siffmath.fluorescence import (
    FluorescenceTrace, dFoF, photon_counts, # noqa: F401
)
from siffpy.siffmath.utils import Timeseries # noqa: F401
import siffpy.siffmath.fluorescence as fluorescence

def fluorescence_fcns(print_docstrings : bool = True) -> List[str]:
    """
    List of public functions available from fluorescence
    submodule. Seems a little silly since I can just use
    the __all__ but this way I can also print the
    docstrings.
    """
    fcns = inspect.getmembers(
        fluorescence,
        lambda x: inspect.isfunction(x) and issubclass(inspect.signature(x).return_annotation, FluorescenceTrace)
    )

    print_string = ""

    for member_fcn_info in fcns:
        fcn_name = member_fcn_info[0]
        fcn_call = member_fcn_info[1]
        print_string += f"\033[1m{fcn_name}\033[0m\n\n"
        print_string += f"\t{fcn_name}{inspect.signature(fcn_call)}\n\n"
        print_string += textwrap.indent(str(inspect.getdoc(fcn_call)),"\t\t")
        print_string += "\n\n"

    if print_docstrings:
        print(print_string)
    else:
        return fcns