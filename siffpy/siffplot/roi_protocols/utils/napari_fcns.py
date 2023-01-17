"""
Collection of functions that are otherwise
in the __init__.py file but use napari as
a dependency, so I don't import them unless
the napari import in __init__.py is 
successful. This is to maintain compatibility
whether the user wants to use just HoloViews
and Bokeh or add napari to the mix for better
image interactions.

Most of these functions either:
1) reimplement functions designed to annotate
ROIs from HoloViews objects
or
2) convert HoloViews or napari objects back
and forth.

SCT 12/11/2021
"""
from typing import Callable, Iterable, Union
import numpy as np
import napari
from napari.layers.shapes import Shapes
import holoviews as hv
import holoviews.element.path as hvpath

from siffpy.siffplot.roi_protocols.rois import ROI

__all__ = [
    'get_largest_lines_napari',
    'get_largest_polygon_napari',
    'get_largest_ellipse_napari',
    'holoviews_to_napari_shapes',
    'napari_shapes_to_holoviews',
    'rois_into_shapes_layer',
]

class _NapariLike():
    """
    Class containing metadata required to produce a new
    drawn object on a napari Shapes layer. Surprised napari
    doesn't have a class like this already!
    """
    SHAPETYPES = [
        "line",
        "rectangle",
        "ellipse",
        "path",
        "polygon",
    ]

    def __init__(self, shapelike : Union[hvpath.BaseShape, hvpath.Path, hvpath.Polygons] = None):
        """
        Create a _NapariLike class that contains all required information to pass to the add
        method of a napari Shapes layer. Can parse a HoloViews element as well.
        """
        self.data : np.ndarray = None
        self._shape_type : str = None
        self.edge_width : float = 1.0,
        self.edge_color = "#FFFFFF",
        self.face_color = None,
        self.z_index : int = None,
        self._slice : int = None

        if not (shapelike is None):
            self.convert_hv(shapelike)
    
    def convert_hv(self, shapelike : Union[hvpath.BaseShape, hvpath.Path, hvpath.Polygons]):
        """ Converts a holoviews element into a _NapariLike """
        typename = type(shapelike).__name__
        if typename == 'Polygons':
            self.shape_type = 'polygon'
            self._to_polygon(shapelike)
            return
        if typename == 'Path':
            self.shape_type = 'path'
            self._to_path(shapelike)
            return
        if typename == 'Box':
            self.shape_type = 'rectangle'
            self._to_rectangle(shapelike)
            return
        if typename == 'Ellipse':
            self.shape_type = 'ellipse'
            self._to_ellipse(shapelike)
            return
        raise ValueError("Invalid HoloViews shapelike object passed.")

    def _to_polygon(self, shapelike : hvpath.Polygons):
        """
        Called on hv.Polygons
        It's just the y and x points. HoloViews separates them in an OrderedDict,
        and napari stacks them by row.
        """
        self.data = [np.stack((polygon['y'], polygon['x']),axis=-1) for polygon in shapelike.data]

    def _to_path(self, shapelike : hvpath.Path):
        """ Called on hv.Path """
        raise NotImplementedError("Path-to-path conversion in napari_fcns not yet implemented.")

    def _to_rectangle(self, shapelike : hvpath.Box):
        """ Called on hv.Box """
        raise NotImplementedError("Box-to-rectangle conversion in napari_fcns not yet implemented.")

    def _to_ellipse(self, shapelike : hvpath.Ellipse):
        """
        Called on hv.Ellipse.

        HoloViews stores the x and y coordinates of the center, the width and height of the ellipse,
        and the rotation about the center.

        Napari stores the bounding box corners as:
        TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
        (from the viewing perspective),
        or:
        least-x-least-y, 

        To go from HoloViews to napari you:

        Divide the width and height by 2 to get the corners of an ellipse about the origin.

        Rotate about the origin.

        Shift upwards by the distance to the x and y centers.
        """
        ellipse_bounding_box = np.array([
            [-shapelike.height/2, -shapelike.width/2],
            [-shapelike.height/2, shapelike.width/2],
            [ shapelike.height/2, shapelike.width/2],
            [ shapelike.height/2, -shapelike.width/2]            
        ])

        theta : float = -shapelike.orientation # NOTE THE MINUS SIGN
        ROT = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        ellipse_bounding_box = np.matmul(ROT,ellipse_bounding_box.T).T # rotate and transpose back

        ellipse_bounding_box[:,0] += shapelike.y
        ellipse_bounding_box[:,1] += shapelike.x

        self.data = ellipse_bounding_box

    @property
    def slice_idx(self):
        return self._slice

    @slice_idx.setter
    def slice_idx(self, value : int):
        if not isinstance(value, int):
            raise ValueError("slice_idx must be an int")
        
        def add_z_info(data_array : np.ndarray, z_val : int)->np.ndarray:
            # no z-dimension
            if data_array.shape[-1] == 2:
                newstack = np.hstack(
                    (
                        value * np.ones((data_array.shape[0],1)),
                        data_array
                    )
                )
            # already has a z dimension
            elif data_array.shape[-1] == 3:
                newstack = data_array.copy()
                newstack[:,0] = value
            else:
                raise ValueError("Invalid data array encountered")
            return newstack


        if isinstance(self.data, list):
            self.data = [add_z_info(data_array, value) for data_array in self.data]
        elif isinstance(self.data, np.ndarray):
            self.data = add_z_info(self.data, value)
        else:
            raise ValueError("Invalid data array stored.")

    @property
    def shape_type(self):
        if not ((self._shape_type is None) or (self._shape_type in _NapariLike.SHAPETYPES)):
            raise ValueError(f"Inappropriate shape_type set ({self._shape_type})")
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value : str):
        if not value in _NapariLike.SHAPETYPES:
            self._shape_type = None
            raise ValueError("Invalid shape_type set!")
        self._shape_type = value


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
            if (individual_array.shape[-1] == 3) and np.all(np.round(individual_array[:,0]).astype(int) == slice_idx)
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
                if (individual_array.shape[-1] == 3) and np.all(np.round(individual_array[:,0]).astype(int) == this_slice)
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

def _shapes_to_ellipses(shape_layer : Shapes) -> list[np.ndarray]:
    """
    Takes a napari shapes layer and returns a list of ellipses
    """
    return [shape_layer.data[idx] for idx in range(len(shape_layer.data)) if shape_layer.shape_type[idx] == 'ellipse']

def _shapes_to_lines(shape_layer : Shapes) -> list[np.ndarray]:
    """
    Takes a napari shapes layer and returns a list of lines
    """ 
    return [shape_layer.data[idx] for idx in range(len(shape_layer.data)) if shape_layer.shape_type[idx] == 'line']

def _largest_polygon_tuple_from_viable(polys_list : list[np.ndarray], n_polygons : int = 1) -> tuple[hv.element.path.Polygons,int, int]:
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

def _largest_ellipse_tuple_from_viable(ellipse_list : list[np.ndarray], n_ellipses : int = 1) -> tuple[hv.element.path.Ellipse,int, int]:
    """
    Returns the appropriate ellipse tuple (or list) for get_largest_ellipse from a list of viable
    ellipses as a hv.element.Polygons.

    Ellipse format: [
        [z1, y1, x1],
        [z2, y2, x2],
        [z3, y3, x3],
        [z4, y4, x4]
    ]
    """
    areas = [_polygon_area(ellipse[:,-1], ellipse[:,-2]) for ellipse in ellipse_list] # polygon_area(x, y) is just a scaled version of ellipse area
    roi_idxs = np.argpartition(np.array(areas), -n_ellipses)[-n_ellipses:]
    ret_list = []
    # If there's only one, don't bother returning as a list
    if n_ellipses == 1:
        ellipse_array = ellipse_list[roi_idxs[-1]]
        if ellipse_array.shape[-1] == 2: # 2 dimensional, not 3d
            ret_slice = None
        else: # otherwise, get the z slice of the first point
            ret_slice = int(np.round(ellipse_array[0][0]))
        
        avgs = np.mean(ellipse_array,axis=0)
        
        width = np.sqrt(np.sum((ellipse_array[0,-2:] - ellipse_array[1,-2:])**2))
        height = np.sqrt(np.sum((ellipse_array[1,-2:] - ellipse_array[2,-2:])**2))

        diff1 = ellipse_array[1,-2:] - ellipse_array[0,-2:] 
        theta = -np.arctan(diff1[0]/diff1[1])

        return (
            hv.Ellipse(
                avgs[-1], # center x
                avgs[-2], # center y
                (width, height), # width, height
                orientation = -theta, # counterclockwise rotation from x axis
            ),
            ret_slice,
            roi_idxs[-1]
        )

    #hv.Ellipse(
    #        center_x,
    #        center_y,
    #        (width, height),
    #        orientation = theta
    #    )

    # If there's more than one polygon requested,
    # return a list of polygons
    for idx in roi_idxs:
        ellipse_array = ellipse_list[idx]
        if ellipse_array.shape[-1] == 2:
            ret_slice = None
        else:
            ret_slice = int(np.round(ellipse_array[0][0]))
        
        avgs = np.mean(ellipse_array,axis=0)
        
        width = np.sqrt(np.sum((ellipse_array[0,-2:] - ellipse_array[1,-2:])**2))
        height = np.sqrt(np.sum((ellipse_array[1,-2:] - ellipse_array[2,-2:])**2))

        diff1 = ellipse_array[1,-2:] - ellipse_array[0,-2:] 
        theta = -np.arctan(diff1[0]/diff1[1])
        
        ret_list.append(
            (
                hv.Ellipse(
                    avgs[-1], # center x
                    avgs[-2], # center y
                    (width, height), # width, height
                    orientation = -theta, # counterclockwise rotation from x axis
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
            ret_slice = int(np.round(line_array[0][0]))
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
            ret_slice = int(np.round(line_array[0][0]))
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

def get_largest_ellipse_napari(
        viewer : napari.Viewer,
        shape_layer_name : str = 'ROI shapes',
        slice_idx : int = None,
        n_ellipses : int = 1
    ) -> tuple[hv.element.path.Ellipse, int, int]:
    """
    Expects a napari Viewer, and returns the largest ellipse in the
    shapes layer(s) named + from which slice it came. n_ellipses is the
    number of polygons to return.
    If >1, returns a LIST of tuples, with the 1st tuple being the largest polygon, 
    the next being the next largest polygon, etc. If there
    are fewer polygons than requested, will raise an exception. 

    Returns as Holoviews Ellipse, so that the napari import variation
    issues are resolved.
    """
    if not (hasattr(viewer, 'layers')):
        raise ValueError(f"Argument viewer must be of type napari.Viewer, not {type(viewer)}")

    # get the layer matching the name expected. Throws an
    # error if no such layer.
    poly_layer = next(filter(lambda x: x.name == shape_layer_name, viewer.layers),None) 
    if not type(poly_layer) is Shapes:
        raise TypeError(f"Specified layer {shape_layer_name} is not a layer of polygons (shapes)")
    
    return _slice_idx_parsing(slice_idx, _shapes_to_ellipses, _largest_ellipse_tuple_from_viable, n_ellipses, poly_layer)    

def get_largest_polygon_napari(
        viewer : napari.Viewer, 
        shape_layer_name : str = 'ROI shapes',
        slice_idx : int = None,
        n_polygons : int = 1
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

def rois_into_shapes_layer(
        rois : Iterable[ROI],
        shapes_layer : Shapes
    ):
    """
    Takes an iterable of siffpy ROI objects and draws them as comparable
    objects in the passed napari Shapes layer
    """

    # Iterates through each item, converts it into something napari could understand,
    # and then plots it.
    for roi in rois:
        shapelike = roi.polygon
        naparilike = _NapariLike(shapelike)
        if hasattr(roi, 'slice_idx'):
            naparilike.slice_idx = roi.slice_idx
        if 'fill_color' in roi.plotting_opts:
            naparilike.face_color = roi.plotting_opts['fill_color']
        if 'fill_alpha' in roi.plotting_opts:
            alpha = roi.plotting_opts['fill_alpha']
            facecolor = naparilike.face_color
            if isinstance(facecolor, list):
                if len(facecolor) == 3:
                    facecolor.append(alpha)
                if len(facecolor) == 4:
                    facecolor[3] = alpha
            if isinstance(facecolor, str) or (facecolor is None):
                if isinstance(alpha,str):
                    alpha_to_hex = alpha
                if isinstance(alpha, float):
                    alpha = max(1.0, alpha)
                    alpha = int(alpha*255)
                    alpha_to_hex = "{0:x}".format(alpha)
                if isinstance(alpha, int):
                    alpha = max(255, alpha)
                    alpha_to_hex = "{0:x}".format(alpha)
                if len(facecolor) == 7: # RGB
                    facecolor = facecolor + alpha_to_hex
                if len(facecolor) == 9: # RGBA
                    facecolor = list(facecolor)
                    facecolor[7:9] = alpha_to_hex
                    facecolor = ''.join(facecolor)
            naparilike.face_color = facecolor

        shapes_layer.add(
            naparilike.data,
            shape_type = naparilike.shape_type,
            edge_width = naparilike.edge_width,
            edge_color = naparilike.edge_color,
            face_color = naparilike.face_color
        )