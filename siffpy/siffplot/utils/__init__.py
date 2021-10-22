from itertools import tee
import numpy as np

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..., (sn-2, sn-1), (sn-1, sn)"
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

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