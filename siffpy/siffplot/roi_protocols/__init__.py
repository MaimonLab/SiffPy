from . import ellipsoid_body, fan_shaped_body, protocerebral_bridge

# Default method for each brain region of interest
# Written this way so I can use the same names for
# different brain regions and have them be implemented
# differently.

def eb_rois(method_name : str = None, *args, **kwargs):
    if method_name is None:
        method_name = "fit_ellipse"

    if not callable(getattr(ellipsoid_body, method_name)):
        raise ValueError(f"Ellipsoid body ROI fitting method {method_name} does not exist!")
    
    fit_method = getattr(ellipsoid_body, method_name)
    return fit_method(*args, **kwargs)

def fb_rois(method_name : str = None, *args, **kwargs):
    if method_name is None:
        raise NotImplementedError()

    if not callable(getattr(fan_shaped_body, method_name)):
        raise ValueError(f"Fan-shaped body ROI fitting method {method_name} does not exist!")
    
    fit_method = getattr(fan_shaped_body, method_name)
    return fit_method(*args, **kwargs)

def pb_rois(method_name : str = None, *args, **kwargs):
    if method_name is None:
        raise NotImplementedError()

    if not callable(getattr(protocerebral_bridge, method_name)):
        raise ValueError(f"Protocerebral bridge ROI fitting method {method_name} does not exist!")
    
    fit_method = getattr(protocerebral_bridge, method_name)
    return fit_method(*args, **kwargs)

def ROI_extraction_methods() -> list[str]:
    return [""]



class ROI():
    """
    Class for an ROI. Contains information about bounding box, brain region to which
    this ROI belongs, method produced to extract this ROI, and probably information about
    how to use it for computations
    """
    def __init__(self):
        pass