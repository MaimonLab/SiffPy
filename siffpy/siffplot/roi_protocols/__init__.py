import re
import logging
import inspect, textwrap

from siffpy.siffplot.roi_protocols.rois import ROI
from siffpy.siffplot.roi_protocols import ellipsoid_body, fan_shaped_body, protocerebral_bridge, noduli, generic

# Default method for each brain region of interest
# Written this way so I can use the same names for
# different brain regions and have them be implemented
# differently.

def region_name_proper(region : str = None) -> str:
    for (key, value) in REGIONS.items():
        if str_in_list(region, value['alias_list']):
            return key

    raise ValueError(f"Region name {region} unknown")

def roi_protocol(region : str, method_name : str, *args, **kwargs):
    """
    Calls the appropriate protocol for the region specified, with provided args and kwargs.
    
    If method_name is None, then the default method for that region will be provided.
    
    ROI protocols, as a general rule, take the siffreader's reference frames and the 
    polygon source (generally a napari Viewer, a NapariInterface, or an annotation_dict),
    as the first two args. But this is not strictly necessary! roi_protocol permits
    any *args and **kwargs arrangement.
    """

    for (key, value) in REGIONS.items():
        
        if str_in_list(region,value['alias_list']): # if it's a valid region name
            
            if method_name is None: # no method is specified, use the default
                method_name = value['default_fcn']
                logging.warn(
                    f"Using default ROI method '{method_name}' for {region_name_proper(region)}." +
                    "For a list of alternatives, call siffpy.siffplot.roi_protocols.ROI_extraction_methods()"
                )

            if not callable(getattr(value['module'], method_name)): # check that the method IS callable
                raise ValueError(f"No ROI fitting method {method_name} in ROI module {value['module']}")
            
            fit_method = getattr(value['module'], method_name)

            return fit_method(*args, **kwargs)
    
    raise ValueError(f"Unable to find region with alias {region}")

def ROI_extraction_methods(print_output : bool = True) -> dict[str, list[str]]:
    """
    Prints each ROI method and its docstring, organized by region.

    RETURNS
    -------
    Returns a dict whose keys are the names of available regions, and whose values
    are each a list of strings, with each string a name of a method usable. So,
    for example, it may return something like:

    {
        'region A' : [
                        'method_1A',
                        'method_2A',
                        ...,
                        'method_nA'
                    ],
        'region B' : [
                        'method_1B',
                        'method_2B',
                        ...,
                        'method_nB'
                    ],
                    
        ...
    }        
    
    """
    print_string = f""
    ret_stringdict = {}
    for (region_name, region_info) in REGIONS.items():
        print_string += f"\033[1m{region_name}\033[0m\n\n"
        memberfcns = inspect.getmembers(region_info['module'], inspect.isfunction)
        ret_stringdict[region_name] = []
        
        for member_fcn_info in memberfcns:
            fcn_name = member_fcn_info[0]
            fcn_call = member_fcn_info[1]
            print_string += f"\t{fcn_name}\n\n"
            print_string += f"\t{fcn_name}{inspect.signature(fcn_call)}\n\n"
            print_string += textwrap.indent(str(inspect.getdoc(fcn_call)),"\t\t")
            print_string += "\n\n"
            
            ret_stringdict[region_name].append(fcn_name)

    if print_output:
        print(print_string)
    return ret_stringdict

def str_in_list(string : str, target_list : list[str]) -> bool:
    """ Cleaner one-liner for case-insensitive matching within a list of strings """
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

NODULI = [
    'no',
    'noduli',
    'nodulus',
    'nod'
]

GENERIC = [
    'generic'
]

REGIONS = {
    'Ellipsoid body' : 
        {
            'alias_list'  : ELLIPSOID_BODY,
            'module'      : ellipsoid_body,
            'default_fcn' : 'use_ellipse'
        },
    
    'Fan-shaped body' : 
        {
            'alias_list'  : FAN_SHAPED_BODY,
            'module'      : fan_shaped_body,
            'default_fcn' : 'outline_fan'
        },
    
    'Protocerebral bridge' : 
        {
            'alias_list'  : PROTOCEREBRAL_BRIDGE,
            'module'      : protocerebral_bridge,
            'default_fcn' : 'dummy_method'
        },

    'Noduli' : 
        {
            'alias_list'  : NODULI,
            'module'      : noduli,
            'default_fcn' : 'hemispheres'
        },

    'Generic' : 
        {
            'alias_list'  : GENERIC,
            'module'      : generic,
            'default_fcn' : 'outline_roi'
        }
}