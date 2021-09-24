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