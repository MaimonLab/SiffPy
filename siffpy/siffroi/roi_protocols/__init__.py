import re
import logging
import inspect, textwrap

from siffpy.siffroi.roi_protocols.roi_protocol import ROIProtocol
from siffpy.siffroi.roi_protocols.rois import ROI
from siffpy.siffroi.roi_protocols.utils.regions import RegionEnum, Region
from siffpy.siffroi.roi_protocols import ellipsoid_body, fan_shaped_body, protocerebral_bridge, noduli, generic

REGIONS = [
    Region(
        ['eb','ellipsoid body','ellipsoid', 'Ellipsoid body'],
        ellipsoid_body,
        'Use ellipse',
        RegionEnum.ELLIPSOID_BODY
    ),
    Region(
        ['fb','fsb','fan-shaped body','fan shaped body','fan', 'Fan-shaped body'],
        fan_shaped_body,
        'Outline fan',
        RegionEnum.FAN_SHAPED_BODY
    ),
    Region(
        ['pb','pcb','protocerebral bridge','bridge', 'Protocerebral bridge'],
        protocerebral_bridge,
        'Fit von Mises',
        RegionEnum.PROTOCEREBRAL_BRIDGE
    ),
    Region(
        ['no','noduli','nodulus','nod', 'Noduli'],
        noduli,
        'dummy_method',
        RegionEnum.NODULI
    ),
    Region(
        ['generic'],
        generic,
        'dummy_method',
        RegionEnum.GENERIC
    )
]

# Default method for each brain region of interest
# Written this way so I can use the same names for
# different brain REGIONS and have them be implemented
# differently.

def roi_protocol(region_str : str, method_name : str, *args, **kwargs):
    """
    Calls the appropriate protocol for the region specified, with provided args and kwargs.
    
    If method_name is None, then the default method for that region will be provided.
    
    ROI protocols, as a general rule, take the siffreader's reference frames and the 
    polygon source (generally a napari Viewer, a NapariInterface, or an annotation_dict),
    as the first two args. But this is not strictly necessary! roi_protocol permits
    any *args and **kwargs arrangement.
    """

    for region in REGIONS:
        
        if str_in_list(region_str,region.alias_list) or region_str == region.region_enum.value: # if it's a valid region name
            
            if method_name is None: # no method is specified, use the default
                method_name = region.default_fcn
                logging.warn(
                    f"Using default ROI method '{method_name}' for {region.region_enum.value}." +
                    "For a list of alternatives, call siffpy.siffplot.roi_protocols.ROI_extraction_methods()"
                )

            if not any(method_name == method.name for method in region.functions ): # check that the method IS callable
                raise ValueError(f"No ROI fitting method {method_name} for ROI of type {region.region_enum.value}")
            
            fit_method = next(method.func for method in region.functions if method.name == method_name)

            return fit_method(*args, **kwargs)
    
    raise ValueError(f"Unable to find region with alias {region}")

def ROI_extraction_methods(print_output : bool = True) -> dict[str, list[str]]:
    """
    Prints each ROI method and its docstring, organized by region.

    RETURNS
    -------
    Returns a dict whose keys are the names of available REGIONS, and whose values
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

    def print_region(print_string : str, ret_stringdict : dict, region : Region):
        """ Local function for formatting the strings for a region """
        print_string += f"\033[1m{region.region_enum.value}\033[0m\n\n"
        memberfcns = region.functions
        ret_stringdict[region.region_enum.value] = []
        for member_fcn in memberfcns:
            print_string += f"\t{member_fcn.name}\n\n"
            print_string += f"\t{member_fcn.func}{inspect.signature(member_fcn.func)}\n\n"
            print_string += textwrap.indent(str(inspect.getdoc(member_fcn.func)),"\t\t")
            print_string += "\n\n"
            
            ret_stringdict[region.region_enum.value].append(member_fcn.name)
        return print_string, ret_stringdict

    for region in REGIONS:
        print_string, ret_stringdict = print_region(print_string, ret_stringdict, region)

    if print_output:
        print(print_string)

    return ret_stringdict

def str_in_list(string : str, target_list : list[str]) -> bool:
    """ Cleaner one-liner for case-insensitive matching within a list of strings """
    return bool(re.match('(?:% s)' % '|'.join(target_list), string, re.IGNORECASE)) 
