# parameters describing a fictrac ball


DEFAULT_PARAMS = {
    'radius'   : 3,
    'units'    : 'mm',
    'axis'     : 'free',
    'material' : 'foam'
}

class BallParams():

    def __init__(self, **kwargs):

        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

        for default_param in DEFAULT_PARAMS:
            if not hasattr(self, default_param):
                setattr(self, default_param, DEFAULT_PARAMS[default_param])
        
        