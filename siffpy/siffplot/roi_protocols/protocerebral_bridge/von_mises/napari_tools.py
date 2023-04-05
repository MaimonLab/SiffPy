from magicgui import widgets
import napari

class CorrelationWindow():

    # I know this must be bad, because I'd be doing circular
    # import stuff to type hint this.
    def __init__(self, segmentation_widget):
        self.viewer = napari.Viewer(
            title = "Von Mises Corrrelation Analysis"
        )

        done_button = widgets.PushButton(
            text = 'Done!'
        )

        done_button.clicked.connect(
            lambda *x: segmentation_widget.events.extraction_initiated() 
        )
        self.viewer.window.add_dock_widget(
            done_button,
        )