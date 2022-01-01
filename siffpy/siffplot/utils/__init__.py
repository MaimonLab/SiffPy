"""
Convenience functions for Plotter type classes
"""
from itertools import tee
import inspect

import numpy as np
import holoviews as hv

from ...siffmath import fluorescence

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..., (sn-2, sn-1), (sn-1, sn)"
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

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

def split_headings_to_dict(x_var : np.ndarray, y_var : np.ndarray, xlabel : str ='x', ylabel : str ='y')->list[dict]:
    """
    Returns a list of dicts that can be passed to a Holoviews Path object.

    Splits circular variables up whenever there's a shift > pi in magnitude.

    ARGUMENTS
    ---------
    
    x_var : 
        
        The independent variable (usually the kdim) to pass to Holoviews

    y_var :

        The dependent variable (usually to be the vdim) to pass to Holoviews

    xlabel : str

        What the key for the x_var should be

    ylabel : str

        What the key for the y_var should be

    RETURNS
    -------

    split : list[dict]

        A list of dicts corresponding to individual line segments before they need to be
        made disjoint. Schematized as:

        [
            {
                xlabel : consecutive_x_coords,
                ylabel : consecutive_y_coords
            }
        ]

    """
    if np.where(np.abs(np.diff(y_var))>=np.pi)[0].shape[0] == 0 :
        return [ {xlabel : x_var, ylabel : y_var} ]

    split = [
                {
                    xlabel : x_var[(start+1):end],
                    ylabel : (y_var[(start+1):end])
                }
                for start, end in pairwise(np.where(np.abs(np.diff(y_var))>=np.pi)[0]) # segments whenever the rotation is of magnitude > np.pi
    ]
    return split

def split_headings_to_list(heading : np.ndarray)->list[np.ndarray]:
    """
    Splits circular variables up whenever there's a shift > pi in magnitude
    and returns a list of each of the fragments

    ARGUMENTS
    ---------
    
    heading : np.ndarray 
        
        The independent variable (usually the kdim) to pass to Holoviews

    RETURNS
    -------

    split : list[np.ndarray]

        A list of arrays corresponding to individual line segments before they need to be
        made disjoint. Schematized as:

    """
    if np.where(np.abs(np.diff(heading))>=np.pi)[0].shape[0] == 0 :
        return [ heading ]

    split = [
            heading[(start+1):end]
            for start, end in pairwise(np.where(np.abs(np.diff(heading))>=np.pi)[0]) # segments whenever the rotation is of magnitude > np.pi
    ]
    return split

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
    fcns = inspect.getmembers(fluorescence, inspect.isfunction)
    if print_docstrings:
        return ["\033[1m" + fcn[0] + ":\n\n\t" + str(inspect.getdoc(fcns[1])) + "\033[0m" for fcn in fcns if fcn[0] in fluorescence.__dict__['__all__']] 
    return [fcn[0] for fcn in fcns if fcn[0] in fluorescence.__dict__['__all__']]


def fifth_percentile(rois : np.ndarray) -> np.ndarray:
    sorted_array = np.sort(rois,axis=1)
    return sorted_array[:, rois.shape[0]//20]