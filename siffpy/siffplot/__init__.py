import logging
import os

import holoviews as hv
from holoviews import opts

BOKEH = False
try:
    import bokeh
    from bokeh.plotting import Figure
    from bokeh.models.layouts import LayoutDOM
    from bokeh.io import export_svg, export_svgs
    BOKEH = True
except ImportError:
    pass

LATEX = False
if BOKEH:
    LATEX = (bokeh.__version__ >= '2.4') and (hv.__version__ > '1.14.6')


from siffpy.siffplot.siffplotter import SiffPlotter
from siffpy.siffplot.siffvisualizer import SiffVisualizer
from siffpy.siffplot.plotters import *
from siffpy.siffplot.visualizers import *
from siffpy.siffplot.roi_protocols import ROI_extraction_methods, ROI

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
            
        logging.warning(
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

def siffpy_export_svgs(fig : hv.Layout, filename : str = None):
    """
    Takes a Holoviews Layout object and exports its individual
    components as svgs in a folder

    Arguments
    --------

    fig : hv.Layout

        A HoloViews Layout object that can't be simply export_svg'd.

    filename : str

        A path to save the file in
    """
    if not BOKEH:
        raise ImportError("Bokeh not available on this system")

    try:
        fig = hv.render(fig)
    except AttributeError:
        # Presumes this means fig is already a Bokeh class
        # or a tuple and I can safely iter
        if hasattr(fig, '__iter__'):
            figname = fig.__class__.__name__
            dirroot, ext = os.path.splitext(filename)
            if figname in ['list', 'tuple']:
                newpath = dirroot
            else:
                newpath = os.path.join(dirroot,figname)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            for subelement in fig:
                try:
                    filename = os.path.join(newpath, subelement.__class__.__name__)
                    if hasattr(subelement, 'id'):
                        filename += f"_id_{subelement.id}"
                    siffpy_export_svgs(subelement, filename = filename+ext)
                except NotFigureError:
                    pass

    if isinstance(fig, Figure):
        fig.output_backend = 'svg'
        export_svgs(fig, filename = filename)
        return

    if isinstance(fig, LayoutDOM):
        if not hasattr(fig, 'children'):
            return
        
        figname = fig.__class__.__name__
        dirroot, ext = os.path.splitext(filename)
        newpath = os.path.join(dirroot,figname)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        for subelement in fig.children:
            try:
                filename = os.path.join(newpath, subelement.__class__.__name__)
                if hasattr(subelement, 'id'):
                    filename += f"_id_{subelement.id}"
                siffpy_export_svgs(subelement, filename = filename+ext)
            except NotFigureError:
                pass
        return

    raise NotFigureError(f"Argument {fig} is not a `Bokeh` element nor a renderable `Holoviews` object.")

class NotFigureError(ValueError):
    """ Special error for not being a usable figure for a method """