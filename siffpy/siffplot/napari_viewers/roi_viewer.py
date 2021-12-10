import napari

class ROIViewer(napari.Viewer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, title='ROI Viewer', axis_labels=('Plane'), **kwargs)