"""
Collection of functions that are otherwise
in the __init__.py file but use napari as
a dependency, so I don't import them unless
the napari import in __init__.py is 
successful. This is to maintain compatibility
whether the user wants to use just HoloViews
and Bokeh or add napari to the mix for better
image interactions.

SCT 12/11/2021
"""
from typing import Callable, Iterable
import numpy as np
import napari
from napari.layers.shapes import Shapes
import holoviews as hv

__all__ = [
    'get_largest_lines_napari',
    'get_largest_polygon_napari',
    'holoviews_to_napari_shapes',
    'napari_shapes_to_holoviews'
]

#######################
### PRIVATE METHODS ###
#######################

def _slice_idx_parsing(slice_idx, shapes_to_array_fcn : Callable, ret_tuple_fcn : Callable, n_returned : int, shapes_layer : Shapes):
    """
    Generic operation used by several functions here. Parses the argument
    slice_idx, which could be of type int, None, or list[int]. Based on which it is,
    returns either a tuple, a list of tuples, or a list of list of tuples of holoviews
    elements, determined by ret_tuple_fcn.

    Returns None (rather than a tuple) for cases where no eligible shapes are extracted.
    """

    list_of_arrays = shapes_to_array_fcn(shapes_layer) # list of numpy arrays of points

    if slice_idx is None:
        # Get the desired object irrespective of slice
        if len(list_of_arrays) == 0:
            return None
        return ret_tuple_fcn(list_of_arrays, n_returned) # all polygons are viable!
    
    if type(slice_idx) is int:
        # Take only the ones in the correct slice plane
        slice_arrays = [
            individual_array
            for individual_array in list_of_arrays
            if (individual_array.shape[-1] == 3) and np.all(individual_array[:,0] == slice_idx)
        ]
        if len(slice_arrays) == 0:
            return None
        return ret_tuple_fcn(slice_arrays, n_returned)

    if type(slice_idx) is list:
        ret_list = []
        for this_slice in slice_idx:
            try:
                this_slice = int(this_slice)
            except:    
                raise TypeError("At least one element of argument slice_idx cannot be cast to type int")

            slice_arrays = [
                individual_array
                for individual_array in list_of_arrays
                if (individual_array.shape[-1] == 3) and np.all(individual_array[:,0] == this_slice)
            ]

            if len(slice_arrays) == 0:
                ret_list.append(None)
            else:
                ret_list.append(ret_tuple_fcn(slice_arrays, n_returned))

        return ret_list
    
    raise TypeError(f"Argument for slice_idx is {slice_idx}, not of type int, list, or None")

def _polygon_area(x_coords : np.ndarray, y_coords : np.ndarray) -> float:
    """ Shoelace method to compute area"""
    return 0.5*np.abs(
        np.sum(x_coords*np.roll(y_coords,1)-y_coords*np.roll(x_coords,1))
    )

def _shapes_to_polys(shape_layer : Shapes) -> list[np.ndarray]:
    """
    Takes a napari shapes layer and returns a list of polygons
    """
    return [shape_layer.data[idx] for idx in range(len(shape_layer.data)) if shape_layer.shape_type[idx] == 'polygon']

def _shapes_to_lines(shape_layer : Shapes) -> list[np.ndarray]:
    """
    Takes a napari shapes layer and returns a list of lines
    """ 
    return [shape_layer.data[idx] for idx in range(len(shape_layer.data)) if shape_layer.shape_type[idx] == 'line']

def _largest_polygon_tuple_from_viable(polys_list, n_polygons : int = 1) -> tuple[hv.element.path.Polygons,int, int]:
    """
    Returns the appropriate polygon tuple (or list) for get_largest_polygon from a list of viable
    polygons as a hv.element.Polygons.
    """
    areas = [_polygon_area(poly[:,-1], poly[:,-2]) for poly in polys_list] # polygon_area(x, y)
    roi_idxs = np.argpartition(np.array(areas), -n_polygons)[-n_polygons:]
    ret_list = []
    # If there's only one, don't bother returning as a list
    if n_polygons == 1:
        poly_array = polys_list[roi_idxs[-1]]
        if poly_array.shape[-1] == 2: # 2 dimensional, not 3d
            ret_slice = None
        else: # otherwise, get the z slice of the first point
            ret_slice = int(poly_array[0][0])
        return (
            hv.Polygons(
                {('y','x'):poly_array[:,-2:]}
            ),
            ret_slice,
            roi_idxs[-1]
        )

    # If there's more than one polygon requested,
    # return a list of polygons
    for idx in roi_idxs:
        poly_array = polys_list[idx]
        if poly_array.shape[-1] == 2:
            ret_slice = None
        else:
            ret_slice = int(poly_array[0][0])
        ret_list.append(
            (
                hv.Polygons(
                    {('y','x'):poly_array[:,-2:]}
                ),
                ret_slice,
                idx
            )
        )
    return ret_list

def _largest_lines_tuple_from_viable(
    line_list : list[np.ndarray], n_lines : int = 1
    )->list[hv.element.path.Path, int, int]:
    """
    Extracts the longest lines and returns them as tuples
    with the first element as a hv.element.Path as desired by get_largest_lines
    """
    lengths = [(line[0,-1]-line[1,-1])**2 + (line[0,-2] + line[1,-2])**2 for line in line_list] # ignoring sqrt
    roi_idxs = np.argpartition(np.array(lengths), -n_lines)[-n_lines:]
    ret_list = []
    # If there's only one, don't bother returning as a list
    if n_lines == 1:
        line_array = line_list[roi_idxs[-1]]
        if line_array.shape[-1] == 2:
            ret_slice = None
        else:
            ret_slice = int(line_array[0][0])
        return (
            hv.Path(
                {('y','x'):line_array[:,-2:]}
            ),
            ret_slice,
            roi_idxs[-1]
        )

    # If there's more than one line requested,
    # return a list of lines
    for idx in roi_idxs:
        line_array = line_list[idx]
        if line_array.shape[-1] == 2:
            ret_slice = None
        else:
            ret_slice = int(line_array[0][0])
        ret_list.append(
            (
                hv.Path(
                    {('y','x'):line_array[:,-2:]}
                ),
                ret_slice,
                idx
            )
        )
    return ret_list

##########################
### IMPORTABLE METHODS ###
##########################
def get_largest_lines_napari(
        viewer : napari.Viewer,
        shape_layer_name : str = 'ROI shapes',
        slice_idx : int = None,
        n_lines = 2
    ) -> tuple[hv.element.Path, int, int]:
    """
    Extracts the longest line type shapes from the given
    napari Viewer and layer. 
    """
     # get the layer matching the name expected. Throws an
    # error if no such layer.
    lines_layer = next(filter(lambda x: x.name == shape_layer_name, viewer.layers),None) 
    if not type(lines_layer) is Shapes:
        raise TypeError(f"Specified layer {shape_layer_name} is not a layer with lines (i.e. ShapesLayer)")

    return _slice_idx_parsing(slice_idx, _shapes_to_lines, _largest_lines_tuple_from_viable, n_lines, lines_layer)

def get_largest_polygon_napari(
        viewer : napari.Viewer, 
        shape_layer_name : str = 'ROI shapes',
        slice_idx : int = None,
        n_polygons = 1
    ) -> tuple[hv.element.path.Polygons,int, int]:
    """
    Expects a napari Viewer, and returns the largest polygon in the
    shapes layer(s) named + from which slice it came. n_polygons is the
    number of polygons to return.
    If >1, returns a LIST of tuples, with the 1st tuple being the largest polygon, 
    the next being the next largest polygon, etc. If there
    are fewer polygons than requested, will raise an exception. 

    Returns as Holoviews polygons, so that the napari import variation
    issues are resolved.
    """

    # get the layer matching the name expected. Throws an
    # error if no such layer.
    poly_layer = next(filter(lambda x: x.name == shape_layer_name, viewer.layers),None) 
    if not type(poly_layer) is Shapes:
        raise TypeError(f"Specified layer {shape_layer_name} is not a layer of polygons (shapes)")

    return _slice_idx_parsing(slice_idx, _shapes_to_polys, _largest_polygon_tuple_from_viable, n_polygons, poly_layer)

def holoviews_to_napari_shapes(polygons : Iterable[hv.Element], properties : Iterable[dict] = None)-> Shapes:
    """
    Takes a iterable of hv.Elements and transfers them (and/or their contained
    polygons or other elements, if each element has more than one in it), one by one,
    into a single napari Shapes layer, which is returned. The properties can be provided
    or generated for you. By default TODO: FINISH DECIDING WHAT THE DEFAULT WILL DO.

    Parameters
    ----------

    polygons : Iterable[hv.Elements]

        An iterable of elements, the points of which will each be used to produce new napari Shapes.
        HoloViews elements are mapped to napari Shapes as follows:

            - Polygons -> Polygons
            - 
    """
    raise NotImplementedError()

def napari_shapes_to_holoviews(shapesLayer : Shapes, inherit_properties = True)-> Iterable[hv.Element]:
    """
    Takes a napari Shapes layer and transfers each of its shapes into an iterable
    of holoviews Elements. The properties can be inherited or default properties can be
    provided (if inherited, they become opts). By default TODO: FINISH DECIDING WHAT THE DEFAULT WILL DO.

    Parameters
    ----------

    shapesLayer : napari.layers.Shapes

        A napari Shapes layer whose data elements will be mapped into holoviews Elements.
        napari Shapes are mapped to holoviews Elements as follows:

            - Polygons -> Polygons
            - Line -> Paths
            - Ellipse -> Ellipse
            - Rectangle -> Rectangle
    """
    ret_list = []

    # Iterate through the shapes in the shapes layer and convert
    # each to a holoviews Element
    raise NotImplementedError()
    if len(ret_list) == 0:
        return None