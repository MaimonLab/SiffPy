import re

from . import ellipsoid_body, fan_shaped_body, protocerebral_bridge

# Default method for each brain region of interest
# Written this way so I can use the same names for
# different brain regions and have them be implemented
# differently.

def region_name_proper(region : str = None) -> str:
    if str_in_list(region,PROTOCEREBRAL_BRIDGE):
        return "Protocerebral bridge"

    if str_in_list(region, FAN_SHAPED_BODY):
        return "Fan-shaped body"

    if str_in_list(region, ELLIPSOID_BODY):
        return "Ellipsoid body"

    return "Unknown"

def roi_protocol(region : str = None, method_name : str = None, *args, **kwargs):
    if str_in_list(region,PROTOCEREBRAL_BRIDGE):
        return pb_rois(
            method_name,
            *args,
            **kwargs
        )

    if str_in_list(region,FAN_SHAPED_BODY):
        return fb_rois(
            method_name,
            *args,
            **kwargs
        )

    if str_in_list(region,ELLIPSOID_BODY):
        return eb_rois(
            method_name,
            *args,
            **kwargs
        )

    return None

def eb_rois(method_name : str = None, *args, **kwargs):
    """ Allows region-specific default info """
    if method_name is None:
        method_name = "fit_ellipse"

    if not callable(getattr(ellipsoid_body, method_name)):
        raise ValueError(f"Ellipsoid body ROI fitting method {method_name} does not exist!")
    
    fit_method = getattr(ellipsoid_body, method_name)
    return fit_method(*args, **kwargs)

def fb_rois(method_name : str = None, *args, **kwargs):
    """ Allows region-specific default info """
    if method_name is None:
        raise NotImplementedError()

    if not callable(getattr(fan_shaped_body, method_name)):
        raise ValueError(f"Fan-shaped body ROI fitting method {method_name} does not exist!")
    
    fit_method = getattr(fan_shaped_body, method_name)
    return fit_method(*args, **kwargs)

def pb_rois(method_name : str = None, *args, **kwargs):
    """ Allows region-specific default info """
    if method_name is None:
        raise NotImplementedError()

    if not callable(getattr(protocerebral_bridge, method_name)):
        raise ValueError(f"Protocerebral bridge ROI fitting method {method_name} does not exist!")
    
    fit_method = getattr(protocerebral_bridge, method_name)
    return fit_method(*args, **kwargs)

def ROI_extraction_methods() -> list[str]:
    """ TODO: RETURN LIST OF METHODS AVAILABLE FOR EACH ROI TYPE """
    return [""]

def str_in_list(string : str, target_list : list) -> bool:
    """ Cleaner one-liner """
    return bool(re.match('(?:% s)' % '|'.join(target_list), string, re.IGNORECASE)) 

ELLIPSOID_BODY = [
    'eb',
    'ellipsoid body',
    'ellipsoid'
]

FAN_SHAPED_BODY = [
    'fb',
    'fsb',
    'fan-shaped body',
    'fan shaped body',
    'fan'
]

PROTOCEREBRAL_BRIDGE = [
    'pb',
    'pcb',
    'protocerebral bridge',
    'bridge'
]