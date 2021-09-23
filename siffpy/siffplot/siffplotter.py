import functools, re, os, pickle, logging

import numpy as np
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

from ..siffpy import SiffReader
from . import roi_protocols

def apply_opts(func):
    """
    Decorator function to apply a SiffPlotter's
    'local_opts' attribute to methods which return
    objects that might want them. Allows this object
    to supercede applied defaults, because this gets
    called with every new plot.
    """
    def local_opts(*args):
        if hasattr(args[0],'local_opts'):
            try:
                opts = args[0].local_opts # get the local_opts param from self
                return func(*args).opts(opts)
            except:
                pass # worth a try
        else:
            return func(*args)
    return local_opts

class SiffPlotter():
    """
    TODO: DOCSTRING
    

    KWARGS
    ------

    local_opts : iterable

        Accepts opts keyword with an unpackable iterable used for
        all plots generated by this object
    """

    def __init__(self, siffreader : SiffReader, *args, **kwargs):
        self.siffreader : SiffReader = siffreader

        if 'opts' in kwargs:
            if not isinstance(kwargs['opts'], (list, tuple)):
                TypeError("Argument opts only accepts tuples or lists -- something that can be unpacked")
            self.local_opts = kwargs['opts']

        if self.siffreader.opened:
            self.reference_frames : hv.HoloMap = self.reference_frames_to_holomap()

        if any([file.endswith('.roi') for file in os.listdir(os.path.dirname(self.siffreader.filename))]):
            logging.warning("Found .roi file(s) in directory with open file.\nLoading ROI(s)")
            self.load_rois(path = os.path.dirname(self.siffreader.filename))


    @apply_opts
    def reference_frames_to_holomap(self)->hv.HoloMap:
        """
        If the current siffreader has an opened file,
        looks to see if there are stored reference frames,
        and if so returns a HoloViews HoloMap that allows
        viewing each of them
        """
        if not hasattr(self.siffreader, 'reference_frames'):
            return None
        
        self.ref_ds = hv.Dataset(
            (
                range(self.siffreader.im_params.xsize),
                range(self.siffreader.im_params.ysize), 
                range(self.siffreader.im_params.num_slices),
                self.siffreader.reference_frames
            ),
            ['x','y','z'], 'Intensity'
        )

        ref_holomap = self.ref_ds.to(hv.Image, ['x','y'], 'Intensity', groupby=['z'])
        # hard limits
        ref_holomap.opts(xlim = (0, self.siffreader.im_params.xsize), ylim = (0, self.siffreader.im_params.ysize))
        return ref_holomap

### ROI

    def get_roi_reference_layouts(self, merge : bool = True, polygon_shape : str = 'polygons', **kwargs) -> dict[int, dict]:
        """
        Returns a dict of dicts, the structure of which is as follows:

        dict = {
            z_index (int) :
                {
                    'annotator' : hv.annotator instance for z plane,
                    'layout'    : hv.Layout instance showing the reference frame and polygons
                }
            for z_index in range(num_slices)
        }

        KEYWORD ARGUMENTS
        -----------------

        merge : bool (default True)

            Adds a merged Holoviews Layout object that contains all the reference planes in one Layout,
            each with their own Bokeh toolbar. The key in the returned dict is 'merged', not an int

        polygon_shape : str (default is 'polygons')

            Shape of the ROIs being drawn. Options are:
                polygons
                rectangles
                ellipses
        """
        if not hasattr(self, 'reference_frames'):
            self.reference_frames = self.reference_frames_to_holomap()

        # this is kind of a silly construction, but I think it's
        # more clear to spell everything out little bit by little bit
        # this way. 
        if re.match(r'polygon[s]?', polygon_shape, re.IGNORECASE):
            drawdict = {
                zidx:
                hv.Polygons([])
                for zidx in range(self.siffreader.im_params.num_slices)
            }
        elif re.match(r'rectangle[s]?', polygon_shape, re.IGNORECASE):
            drawdict = {
                zidx:
                hv.Rectangles([])
                for zidx in range(self.siffreader.im_params.num_slices)
            }
        elif re.match(r'ellipse[s]?', polygon_shape, re.IGNORECASE):
            raise NotImplementedError("Ellipse keyword argument not yet implemented (not a native Hv/Bokeh drawer)")
        else:
            raise ValueError(f"Invalid optional argument for polygon_shape:\n\t{polygon_shape}")

        annotators = {
            zidx :
            hv.annotate.instance()
            for zidx in range(self.siffreader.im_params.num_slices)
        }

        annotator_layouts = {
            zidx:
            annotator(self.reference_frames[zidx] * drawdict[zidx]).opts(
                hv.opts.Table(width=0), # hide the annotation table
                hv.opts.Layout(merge_tools=False) # don't share the tools
            )
            for zidx, annotator in annotators.items()
        }

        self.annotation_dict = {
            zidx: {
                'annotator' : annotators[zidx],
                'layout'    : annotator_layouts[zidx]
            }
            for zidx in annotators.keys()
        }

        if merge:
            def merge_plots(a, b):
                if isinstance(a, dict):
                    return (a['layout'] + b['layout']).opts(merge_tools = False)
                else:
                    return (a+b['layout']).opts(merge_tools=False)

            self.annotation_dict['merged'] =  functools.reduce(merge_plots, list(self.annotation_dict.values()))

        return self.annotation_dict

    def draw_rois(self, **kwargs):
        self.get_roi_reference_layouts(**kwargs)
        return self.annotation_dict['merged']

    def roi_to_layout(self):
        l = hv.Layout


    def extract_rois(self, region : str, method_name : str = None, *args, **kwargs) -> None:
        """
        Extract ROIs -- uses a different method for each anatomical region.
        ROIs are stored in a class attribute. Must have drawn at least one
        manual ROI on at least one image stored in the SiffPlotter's annotation dict.

        To learn more about 

        Parameters
        ----------
        
        region : str

            Name of the region of interest. Current protocols are for the FB, EB, and PB.

        method_name : str (optional)

            Which ROI extraction method to use. For a list, call siffplot.ROI_fitting_methods()
        
        Returns
        -------

        None

        """
        if not (
                hasattr(self, 'annotation_dict') and
                any([len(x['annotator'].annotated.data) for x in self.annotation_dict.values() if isinstance(x,dict)])
            ):
            raise AssertionError("Siffplotter object has no hand-annotated ROIs")

        self.region = roi_protocols.region_name_proper(region)
        self.rois = roi_protocols.roi_protocol(
            region,
            method_name,
            self.siffreader.reference_frames,
            self.annotation_dict,
            **kwargs
        )
        
        if self.rois is None:
            raise RuntimeError("No rois extracted -- check method used, images provided, etc.")

    def save_rois(self, path : str = None):
        """
        Saves the rois stored in the self.rois attribute. The default path is in the directory with
        the .siff file.

        Arguments
        ---------
        
        path : str (optional)

            Where to save the ROIs.
        """
        if self.rois is None:
            raise RuntimeError("SiffPlotter object has no rois stored")
        
        if path is None:
            if not self.siffreader.opened:
                raise RuntimeError("Siffreader has no open file, and no alternative path was provided.")
            path = os.path.dirname(self.siffreader.filename)

        path = os.path.join(path, os.path.splitext(os.path.basename(self.siffreader.filename))[0])

        try:
            # if rois is an iterable, save each roi.
            iter(self.rois)
            for roi in self.rois:
                roi.save(path)
        except TypeError:
            # else, just save the one.
            self.rois.save(path)
        except Exception as e:
            raise e

    def load_rois(self, path : str = None):
        """
        Loads rois stored at location 'path'. If no path is specified, then
        it looks for .roi files matching the file name
        """

        if path is None:
            path = os.path.join(
                os.path.dirname(self.siffreader.filename),
                os.path.splitext(os.path.basename(self.siffreader.filename))[0]
            )
        
        roi_files = [os.path.join(path,file) for file in os.listdir(path) if file.endswith('.roi')]

        if len(roi_files) > 1:
            # needs an iterable
            self.rois = []
            for roi in roi_files:
                with open(roi, 'rb') as curr_file:
                    self.rois.append(pickle.load(curr_file))
        else:
            with open(roi_files[0], 'rb') as roi_file:
                self.rois = pickle.load(roi_file)
                
### HEATMAP
    @apply_opts
    def make_heatmap(self, transform_function, **kwargs) -> hv.HoloMap:
        """ Takes a function to apply to the data """
        pass