# Code for ROI extraction from the protocerebral bridge after manual input

from siffpy.siffplot.roi_protocols.utils import PolygonSource
from siffpy.siffplot.roi_protocols import rois

def circle_glomeruli(
        reference_frames : list,
        polygon_source : PolygonSource,
        *args,
        n_glomeruli : int = 16,
        slice_idx : int = None,
        **kwargs
    )-> rois.GlobularMustache:
    """
    Takes the n largest ROIs and stitches them together to form a mustache

    TODO: define the reference direction and use it to order the glomeruli

    Parameters
    ----------

    reference_frames : list of numpy arrays

        The siffreader reference frames to overlay

    polygon_source : napari.Viewer or dict whose values are holoviews annotators

    n_glomeruli : int

        How many glomeruli are circled

    slice_idx : None, int, or list of ints (optional)

        Which slice or slices to extract an ROI for. If None, will take the ROI
        corresponding to the largest polygon across all slices.
    """
    ## Outline:
    #
    #   Just takes the largest polygon and assumes it's fan shaped.
    #
    #   Also looks to see if there's a pair of lines that can be used to guide
    #   segmentation, and if so adds that info to the Fan object produced.
    
    #raise NotImplementedError

    #first check if using holoviews

    largest_polygons, slice_idx, _ = polygon_source.get_largest_polygon(
        slice_idx = slice_idx, n_polygons=n_glomeruli
    )
    source_image = polygon_source.source_image(slice_idx)

    return rois.GlobularMustache(
        None,
        slice_idx = slice_idx,
        image = source_image,
        name = 'Protocerebral bridge',
        globular_glomeruli=largest_polygons,
    )
        
def dummy_method(*args, **kwargs):
    print("I'm just a placeholder!")