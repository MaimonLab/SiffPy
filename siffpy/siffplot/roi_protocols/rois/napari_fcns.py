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
import numpy as np
import napari
from napari.layers.shapes import Shapes
import holoviews as hv

__all__ = [
    'get_largest_polygon_napari'
]

def _polygon_area(x_coords : np.ndarray, y_coords : np.ndarray) -> float:
    """ Shoelace method to compute area"""
    return 0.5*np.abs(
        np.sum(x_coords*np.roll(y_coords,1)-y_coords*np.roll(x_coords,1))
    )

def _shapes_to_polys(shape_layer : Shapes) -> list[np.ndarray]:
    """
    Takes a napari shapes layer and returns a list of polygons

    Arguments
    ---------

    shape_layer : napari.layers.shapes.Shapes

        A single Shapes layer.

    Returns
    -------

    polys : list of numpy.ndarrays
    """
    return [shape_layer.data[idx] for idx in range(len(shape_layer.data)) if shape_layer.shape_type[idx] == 'polygon']

def _shapes_to_lines(shape_layer : Shapes) -> list[np.ndarray]:
    """
    Takes a napari shapes layer and returns a list of lines

    Arguments
    ---------

    shape_layer : napari.layers.shapes.Shapes

        A single Shapes layer.

    Returns
    -------

    liness : list of numpy.ndarrays
    """ 
    return [shape_layer.data[idx] for idx in range(len(shape_layer.data)) if shape_layer.shape_type[idx] == 'line']

def _largest_polygon_tuple_from_viable(polys_list, n_polygons : int = 1) -> tuple[hv.element.path.Polygons,int, int]:
    """
    Returns the appropriate polygon tuple (or list) for get_largest_polygon from a list of viable
    polygons.
    """
    areas = [_polygon_area(poly[:,-1], poly[:,-2]) for poly in polys_list] # polygon_area(x, y)
    roi_idxs = np.argpartition(np.array(areas), -n_polygons)[-n_polygons:]
    ret_list = []
    # If there's only one, don't bother returning as a list
    if n_polygons == 1:
        poly_array = polys_list[roi_idxs[-1]]
        if poly_array.shape[-1] == 2:
            ret_slice = None
        else:
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
    if not type(poly_layer) is napari.layers.shapes.shapes.Shapes:
        raise TypeError(f"Specified layer {shape_layer_name} is not a layer of polygons (shapes)")

    polys = _shapes_to_polys(poly_layer) # list of numpy arrays of points

    if slice_idx is None:
        # Get the largest polygon irrespective of slice
        return _largest_polygon_tuple_from_viable(polys, n_polygons) # all polygons are viable!
    
    if type(slice_idx) is int:
        # Take only the ones in the correct slice plane
        slice_polys = [
            poly
            for poly in polys
            if (poly.shape[-1] == 3) and np.all(poly[:,0] == slice_idx)
        ]

        return _largest_polygon_tuple_from_viable(slice_polys, n_polygons)

    if type(slice_idx) is list:
        ret_list = []
        for this_slice in slice_idx:
            if not type(this_slice) is list:
                raise TypeError("At least one element of argument slice_idx is not of type int")
            slice_polys = [
                poly
                for poly in polys
                if (poly.shape[-1] == 3) and np.all(poly[:,0] == this_slice)
            ]

            ret_list.append(_largest_polygon_tuple_from_viable(slice_polys, n_polygons))

        return ret_list
    
    raise TypeError(f"Argument for slice_idx is {slice_idx}, not of type int, list, or None")