"""
Stored initial guesses that are in the ballpark
of typically-measured parameters for various
FLIM fluorophores.

I'll try to remember to keep adding to this as I record more things.

"""
from .flimparams import Exp, Irf

gCamui = (
    Exp(frac = 0.7, tau = 115),
    Exp(frac = 0.3, tau = 25),
    Irf(tau_offset = 20.0, tau_g = 4.0)    
)

FLIM_AKAR = (
    Exp(frac = 0.636, tau = 117.14),
    Exp(frac = .3633, tau = 39.8358),
    Irf(tau_offset = 30.0, tau_g = 8.0)
)

jRCaMP1b = (
    Exp(frac = 0.6, tau=210.0),
    Exp(frac = 0.4, tau=30.0),
    Irf(tau_offset = 62, tau_g = 3.4)
)

jGCaMP7s = (
    Exp(frac = 0.99, tau = 140.0),
    Irf(tau_offset = 26.0, tau_g = 3.0)
)

def available_fluorophores(dtype : type = list) -> list[str]:
    """ 
    Return available fluorophores with initial conditions for FLIM
    fitting. In truth, I doubt this will matter much for most fluorophores,
    but it's useful for the nonlinear solver to start from a good place.

    INPUTS
    ------

    dtype (optional, type):

        Type of returned object. By default is list. If dict, returns
        the initial condition guesses as well.

    RETURN VALUES
    -------------

    fluorophores (list or dict):

        Either a list of strings or a dict whose keys are strings with
        fluorophore names and whose values are the dicts used for constructing
        a FLIMParams object (or to be passed to 'fit_exp')
    """
    
    if dtype == list:
        list_of_fluorophores = [
            'gCamui',
            'FLIM_AKAR',
            'jRCaMP1b',
            'jGCaMP7s'
        ]

    if dtype == dict:
        list_of_fluorophores = {
            None : None,
            'gCamui' : gCamui,
            'FLIM_AKAR': FLIM_AKAR,
            'jRCaMP1b' : jRCaMP1b,
            'jGCaMP7s' : jGCaMP7s
        }

    return list_of_fluorophores
