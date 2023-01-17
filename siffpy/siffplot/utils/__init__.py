"""
Convenience functions for Plotter type classes
"""
from itertools import tee
import inspect
from functools import wraps

import numpy as np
import holoviews as hv

from siffpy.siffmath import fluorescence
from siffpy.siffplot.utils.exceptions import *
from siffpy.siffplot.utils.enums import *

def apply_opts(func):
    """
    Decorator function to apply a SiffPlotter's
    'local_opts' attribute to methods which return
    objects that might want them. Allows this object
    to supercede applied defaults, because this gets
    called with every new plot. Does nothing if local_opts
    is not defined.
    """
    @wraps(func)
    def local_opts(*args, **kwargs):
        if hasattr(args[0],'_local_opts'):
            try:
                opts = args[0]._local_opts # get the local_opts param from self
                if isinstance(opts, list):
                    return func(*args, **kwargs).opts(*opts)
                if isinstance(opts, dict):
                    return func(*args, **kwargs).opts(opts)
            except:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return local_opts
 
def select_on_tap(pt_array, tapped_points, x, y, x2, y2):
    """
    Add nearest point on singletap, remove nearest point on doubletap.

    Used as the callback for an ROI point selection stream
    """
    
    # Single tap
    if None not in [x, y]:
        # Find nearest point
        dist = np.sum((np.array(pt_array) - np.array((x,y)))**2,axis=1)
        idx = np.argmin(dist)
        # If the list is empty, we can't check if it's already in there
        if not tapped_points:
            tapped_points.append(pt_array[idx])
        # Add it if it's not in the list already
        elif not np.any(np.all(pt_array[idx] == tapped_points,axis = 1)):
            tapped_points.append(pt_array[idx])
        return hv.Points(tapped_points)
    
    elif None not in [x2, y2]:
        if not tapped_points:
            return hv.Points([])
        # Find nearest point in dtapped_points
        dist = np.sum((np.array(tapped_points) - np.array((x2,y2)))**2,axis=1)
        idx = np.argmin(dist)
        # Remove it 
        tapped_points.pop(idx)
        return hv.Points(tapped_points)
    
    if not tapped_points:
        return hv.Points([])
    else:
        return hv.Points(tapped_points)

def bounds_hook(plot, elem):
    plot.state.x_range.bounds = 'auto'
    plot.state.y_range.bounds = 'auto'
    
def arial_hook(plot, elem):
    plot.handles['xaxis'].major_label_text_font='arial'
    plot.handles['xaxis'].major_label_text_font_style = 'normal'
    plot.handles['xaxis'].axis_label_text_font = 'arial'
    plot.handles['xaxis'].axis_label_text_font_style = 'normal'
    plot.handles['xaxis'].minor_tick_line_color = None 
    
    plot.handles['yaxis'].major_label_text_font='arial'
    plot.handles['yaxis'].major_label_text_font_style = 'normal'
    plot.handles['yaxis'].axis_label_text_font = 'arial'
    plot.handles['yaxis'].axis_label_text_font_style = 'normal'
    plot.handles['yaxis'].minor_tick_line_color = None 
    plot.handles['yaxis'].major_tick_line_color = None

def font_hook(plot, elem):
    plot.handles['xaxis'].major_label_text_font='arial'
    plot.handles['xaxis'].major_label_text_font_size='16pt'
    plot.handles['xaxis'].major_label_text_font_style = 'normal'
    plot.handles['xaxis'].axis_label_text_font = 'arial'
    plot.handles['xaxis'].axis_label_text_font_size = '16pt'
    plot.handles['xaxis'].axis_label_text_font_style = 'normal'
    plot.handles['xaxis'].minor_tick_line_color = None 
    
    plot.handles['yaxis'].major_label_text_font='arial'
    plot.handles['yaxis'].major_label_text_font_size='16pt'
    plot.handles['yaxis'].major_label_text_font_style = 'normal'
    plot.handles['yaxis'].axis_label_text_font = 'arial'
    plot.handles['yaxis'].axis_label_text_font_size = '16pt'
    plot.handles['yaxis'].axis_label_text_font_style = 'normal'
    plot.handles['yaxis'].minor_tick_line_color = None 
    plot.handles['yaxis'].major_tick_line_color = None
    plot.handles['yaxis'].axis_line_color = None

def string_names_of_fluorescence_fcns(print_docstrings : bool = False) -> list[str]:
    """
    List of public functions available from fluorescence
    submodule. Seems a little silly since I can just use
    the __all__ but this way I can also print the
    docstrings.
    """
    from siffpy.siffmath.fluorescence import FluorescenceTrace
    fcns = inspect.getmembers(
        fluorescence,
        lambda x: inspect.isfunction(x) and issubclass(inspect.signature(x).return_annotation, FluorescenceTrace)
    )
    if print_docstrings:
        return ["\033[1m" + fcn[0] + ":\n\n\t" + str(inspect.getdoc(fcns[1])) + "\033[0m" for fcn in fcns if fcn[0] in fluorescence.__dict__['__all__']] 
    return [fcn[0] for fcn in fcns if fcn[0] in fluorescence.__dict__['__all__']]