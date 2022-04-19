import logging

from .siffplotter import SiffPlotter
from .siffvisualizer import SiffVisualizer
from .plotters import *
from .visualizers import *
import holoviews as hv
from holoviews import opts

from .roi_protocols import ROI_extraction_methods, ROI

def initialize_holoviews(backend : str = 'bokeh', stylesheet : str = None)->None:
    """
    Calls the HoloViews backend initialization and imports
    some preferred settings. May use a style sheet in the
    future, so I'm going to try to have it remind me every
    time!
    """
    if stylesheet is None:
        def bounds_hook(plot, elem):
            plot.state.x_range.bounds = 'auto'
            plot.state.y_range.bounds = 'auto'
            

        def arial_hook(plot, elem):
            plot.handles['xaxis'].major_label_text_font='arial'
            plot.handles['xaxis'].major_label_text_font_size='16pt'
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

            #plot.handles[]

            plot.handles['xaxis'].major_label_text_font='arial'
            plot.handles['xaxis'].major_label_text_color='#000000'
            plot.handles['xaxis'].major_label_text_font_size='16pt'
            plot.handles['xaxis'].major_label_text_font_style = 'normal'
            plot.handles['xaxis'].axis_label_text_font = 'arial'
            plot.handles['xaxis'].axis_label_text_font_size = '16pt'
            plot.handles['xaxis'].axis_label_text_color='#000000'
            plot.handles['xaxis'].axis_label_text_font_style = 'normal'
            plot.handles['xaxis'].minor_tick_line_color = None 
            plot.handles['xaxis'].major_tick_line_color = None
            plot.handles['xaxis'].axis_line_color = None
            
            plot.handles['yaxis'].major_label_text_font='arial'
            plot.handles['yaxis'].major_label_text_color='#000000'
            plot.handles['yaxis'].major_label_text_font_size='16pt'
            plot.handles['yaxis'].major_label_text_font_style = 'normal'
            plot.handles['yaxis'].axis_label_text_font = 'arial'
            plot.handles['yaxis'].axis_label_text_color='#000000'
            plot.handles['yaxis'].axis_label_text_font_size = '16pt'
            plot.handles['yaxis'].axis_label_text_font_style = 'normal'
            plot.handles['yaxis'].minor_tick_line_color = None 

        def font_both_ax_hook(plot, elem):
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
            
        logging.warn(
            "\n\n Using explicit method initialize_holoviews written into siffplot's __init__.,py."
            "\nBetter to use a style sheet!\n"    
        )
        hv.extension(backend)
        opts.defaults(
            opts.HeatMap(hooks = [font_hook, bounds_hook]),
            opts.Polygons(fill_alpha=0.15),
            opts.Curve(hooks = [font_hook, bounds_hook]),
            opts.Path(hooks = [font_hook, bounds_hook]),
            opts.Image(
                cmap='viridis',
                invert_yaxis = True,
                hooks=[bounds_hook]
            ),
            opts.Points(
                hooks = [font_both_ax_hook, bounds_hook]
            )
        )
    else:
        raise NotImplementedError("Haven't implemented a style sheet yet")