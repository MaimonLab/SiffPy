"""
Stored initial guesses that are in the ballpark
of typically-measured parameters for various
FLIM fluorophores.

I'll try to remember to keep adding to this as I record more things.

BAD STYLE! TODO: CLEAN THIS UP
"""

gCamui = {
            'NCOMPONENTS' : 2,
            'EXPPARAMS' : [
                {'FRAC' : 0.7, 'TAU' : 115},
                {'FRAC' : 0.3, 'TAU' : 25}
            ],
            'CHISQ' : 0.0,
            'T_O' : 20,
            'IRF' :
                {
                    'DIST' : 'GAUSSIAN',
                    'PARAMS' : {
                        'SIGMA' : 4
                    }
                }
        }

FLIM_AKAR = {
            'NCOMPONENTS' : 2,
            'EXPPARAMS' : [
                {'FRAC' : 0.636, 'TAU' : 117.1466836},
                {'FRAC' : 0.3633, 'TAU' : 39.8358}
            ],
            'CHISQ' : 0.0,
            'T_O' : 30,
            'IRF' :
                {
                    'DIST' : 'GAUSSIAN',
                    'PARAMS' : {
                        'SIGMA' : 8.0
                    }
                }
        }

jRCaMP1b = {
            'NCOMPONENTS' : 2,
            'EXPPARAMS' : [
                {'FRAC' : 0.6, 'TAU' : 210.0},
                {'FRAC' : 0.4, 'TAU' : 30.0}
            ],
            'CHISQ' : 0.0,
            'T_O' : 62,
            'IRF' :
                {
                    'DIST' : 'GAUSSIAN',
                    'PARAMS' : {
                        'SIGMA' : 3.4
                    }
                }
        }

jGCaMP7s = {
            'NCOMPONENTS' : 1,
            'EXPPARAMS' : [
                {'FRAC' : 0.99, 'TAU' : 140.0}
            ],
            'CHISQ' : 0.0,
            'T_O' : 26.0,
            'IRF' :
                {
                    'DIST' : 'GAUSSIAN',
                    'PARAMS' : {
                        'SIGMA' : 3.0
                    }
                }
        }

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
