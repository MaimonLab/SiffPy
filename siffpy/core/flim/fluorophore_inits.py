"""
Stored initial guesses that are in the ballpark
of typically-measured parameters for various
FLIM fluorophores.

I'll try to remember to keep adding to this as I record more things.

"""
from typing import Union

from siffpy.core.flim.flimparams import Exp, Irf, FLIMParams

FLUOROPHORE_INITIALIZATIONS = [
    FLIMParams(
        Exp(frac = 0.7, tau = 115),
        Exp(frac = 0.3, tau = 25),
        Irf(tau_offset = 20.0, tau_g = 4.0),
        name = 'green-Camui'
    ),
    FLIMParams(
        Exp(frac = 0.636, tau = 117.14),
        Exp(frac = 0.3633, tau = 39.84),
        Irf(tau_offset = 20.0, tau_g = 4.0),
        name = 'FLIM-AKAR'
    ),
    FLIMParams(
        Exp(frac = 0.6, tau = 210.0),
        Exp(frac = 0.4, tau = 30.0),
        Irf(tau_offset = 20.0, tau_g = 3.4),
        name = 'jRCaMP1b'
    ),
    FLIMParams(
        Exp(frac = 0.99, tau =140.0),
        Irf(tau_offset = 26.0, tau_g = 3.0),
        name = 'jGCaMP7s'
    )
]


def available_fluorophores(dtype : type = list)->Union[dict, list]:
    """
    Returns the available fluorophores for FLIM fitting,
    either as a dict storing the FLIMParams objects or
    simply as a list of the string of names, depending on
    the argument dtype
    
    Arguments
    ----------
    
    dtype : type
        
        The type of object to return. Must be either
        list or dict.
    """
    if dtype == list:
        return [fluoro.name for fluoro in FLUOROPHORE_INITIALIZATIONS]
    elif dtype == dict:
        return {fluoro.name:fluoro for fluoro in FLUOROPHORE_INITIALIZATIONS}
    else:
        raise ValueError("dtype must be either list or dict")
