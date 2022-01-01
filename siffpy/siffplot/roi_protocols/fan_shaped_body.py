# Code for ROI extraction from the fan-shaped body after manual input

from . import rois

def outline_fan(reference_frames : list, polygon_source : dict, *args, **kwargs)-> rois.Fan:
    """
    Takes the largest ROI and assumes it's the outline of the fan-shaped body.

    Optionally looks for two lines to define the edges of the fan for a triangle-based
    segmentation of columns.

    Parameters
    ----------

    reference_frames : list of numpy arrays

        The siffreader reference frames to overlay

    polygon_source : napari.Viewer or dict whose values are holoviews annotators

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
    
    slice_idx = None
    if 'slice_idx' in kwargs:
        if isinstance(kwargs['slice_idx'], int):
            slice_idx = kwargs['slice_idx']
        del kwargs['slice_idx']
    
    if type(polygon_source) is dict: # use holoviews
        annotation_dict = polygon_source
        largest_polygon, slice_idx, _ = rois.get_largest_polygon_hv(annotation_dict, slice_idx = slice_idx)
        source_image = rois.annotation_dict_to_numpy(annotation_dict,slice_idx)

        return rois.Fan(
            largest_polygon,
            slice_idx = slice_idx,
            image = source_image
        )

    else: # using napari
        largest_polygon, slice_idx, _ = rois.get_largest_polygon_napari(polygon_source, shape_layer_name = 'ROI shapes', slice_idx = slice_idx)
        reference_frame_layer = next(filter(lambda x: x.name == 'Reference frames', polygon_source.layers)) # get the layer with the reference frames
        source_image = reference_frame_layer.data[slice_idx, :, :]
        
        fan_lines = rois.get_largest_lines_napari(polygon_source, shape_layer_name = 'ROI shapes', slice_idx=slice_idx, n_lines = 2)

        if not fan_lines is None: # use them if you have them
            return rois.Fan(
                largest_polygon,
                slice_idx = slice_idx,
                image = source_image,
                bounding_paths = fan_lines
            )
        else:
            return rois.Fan(
                largest_polygon,
                slice_idx = slice_idx,
                image = source_image
            )