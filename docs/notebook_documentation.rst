Siffpy: A package for reading ``.siff`` files and performing standard transformations of the data
=================================================================================================

One major downside to collecting FLIM (fluorescence lifetime imaging
microscopy) data is that it does not naturally conform to the structure
of an array: samples with high temporal resolution (like the 5
picoseconds of the PicoQuant MultiHarp) will be very sparse data, with
thousands of possible arrival times per pixel and most of those data
being zeros. And so instead of exporting a standard ``.tiff`` file,
``ScanImage-FLIM`` saves data in the ``.siff`` format, which uses a
``.tiff``-like format to store FLIM data. But because this is not a
standard ``.tiff`` file, it needs its own reader. ``SiffPy`` exists to
extract the data from ``.siff`` files and transform them into ``numpy``
arrays and ``Python`` objects that can be easily piped into standard
workflows.

This page contains examples for some simple workflows that are
constrained entirely to ``SiffPy`` or external pacakges. There are also
other packages that exist with the intention of working with ``SiffPy``
(e.g. ``SiffROI``, ``siff-napari``) that do this job as well.

File I/O
--------

The first thing we need to do, of course, is read a file! The main tool
of ``SiffPy`` is the ``SiffReader`` object, which provides a basic API
for returning ``ndarray`` objects. A ``SiffReader`` can be initialized
with a path to a ``.siff`` file, which will be opened automatically, or
it can be initialized in isolation and a file can be passed later with
the ``open`` function:

::

   sr = SiffReader()

   # collect some user input, other info
   ...

   sr.open(path_to_file)

but the most common use case is as below. File opening is generally
pretty fast (no more than a few seconds for several-GB files), but if
you’re reading data from a mounted server that’s not local, I haven’t
optimized the reader to maximize bandwidth yet and it can be slow.

.. code:: ipython3

    from siffpy import SiffReader
    
    # file_path can be a string or a pathlib.Path object,
    # or anything that can be cast to a pathlib.Path object
    #file_path = 'path/to/file.siff'
    file_path = '/Users/stephen/Desktop/Data/imaging/2023-09-20/SS00238FLIMAKAR/Fly1/BarOnAtTen_1.siff'
    
    sr = SiffReader(file_path)
    
    # Returns a `numpy` array of the photon count (i.e. intensity) data
    # contained in the frames indexed as in the provided `frames`
    # argument.
    first_frames = sr.get_frames(frames = [0,1,2,3])
    
    print(first_frames)


.. parsed-literal::

    [[[0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]
      ...
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]]
    
     [[0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]
      ...
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]]
    
     [[0 0 0 ... 0 0 0]
      [0 0 0 ... 1 0 0]
      [0 0 0 ... 0 0 0]
      ...
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]]
    
     [[0 0 0 ... 0 0 0]
      [0 0 0 ... 1 0 0]
      [0 0 0 ... 0 0 0]
      ...
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]]]


ImParams and figuring out which frames to load
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``SiffReader`` object will now have a ``im_params`` attribute that
points to an ``ImParams`` object, an interface for accessing the
metadata in the ``.siff`` file. Printing the ``ImParams`` object will
report some of this metadata (e.g. the number of frames). Some of it is
stored in the metadata of the ScanImage modules, which can be accessed
like attributes.

.. code:: ipython3

    im_par = sr.im_params
    print(im_par)


.. parsed-literal::

    Image parameters: 
    	ScanImage modules : 
    		base
    		Beams
    		CameraManager
    		Channels
    		ConfigurationSaver
    		CoordinateSystems
    		CycleManager
    		Display
    		FastZ
    		IntegrationRoiManager
    		MotionManager
    		Motors
    		Photostim
    		Pmts
    		RoiManager
    		Scan2D
    		Shutters
    		StackManager
    		TileManager
    		UserFunctions
    		WSConnector
    		WaveformManager
    	_num_frames_from_siffio : 387
    	roi_groups : {'imagingRoiGroup': 
            ROI group Default Imaging ROI Group with
            1 ROI(s):
    
            
            ROI Default Imaging Roi with
            1 scanfield(s):
    
            Scanfield Default Imaging Scanfield with parameters:
    	{'ver': 1, 'classname': 'scanimage.mroi.scanfield.fields.RotatedRectangle', 'name': 'Default Imaging Scanfield', 'UserData': None, 'roiUuid': 'E981838A77ED882E', 'roiUuiduint64': 1.682587431e+19, 'centerXY': [0, 0], 'sizeXY': [2, 2], 'rotationDegrees': 0, 'enable': 1, 'pixelResolutionXY': [256, 256], 'pixelToRefTransform': [[0.0078125, 0, -1.00390625], [0, 0.0078125, -1.00390625], [0, 0, 1]], 'affine': [[2, 0, -1], [0, 2, -1], [0, 0, 1]]}
            
            , 'integrationRoiGroup': 
            ROI group  with
            1 ROI(s):
    
            
            ROI  with
            1 scanfield(s):
    
            
            
            }
    




.. parsed-literal::

    FastZ module: 
    	submodules : {}
    	actuatorLag : 0
    	discardFlybackFrames : True
    	enable : True
    	enableFieldCurveCorr : False
    	errorMsg : 
    	flybackTime : 0.015
    	hasFastZ : True
    	name : SI FastZ
    	numDiscardFlybackFrames : 1
    	position : -30
    	reserverInfo : 
    	userInfo : 
    	volumePeriodAdjustment : -0.0006
    	warnMsg : 
    	waveformType : sawtooth



.. code:: ipython3

    print(im_par.FastZ)

The most useful thing you’ll likely use the ``ImParams`` object to do is
call its framelist functions. These use the ScanImage metadata to
compute which frames in the ``.siff`` file correspond to which parts of
the imaging volume / session. This way you don’t need to figure out
things like what order frames are in, which frames to skip because
they’re flyback, etc. etc. For more information, please check the
``SiffReader`` documentation in the ``API`` section of the docs.

.. code:: ipython3

    # Get the indices of all frames by timepoint (i.e. across all planes, technically
    # slightly separated in time). Note that this example skips frame 6, which
    # in this experiment was a flyback frame
    im_par.flatten_by_timepoints(timepoint_start = 0, timepoint_end = 10)




.. parsed-literal::

    [0,
     1,
     2,
     3,
     4,
     5,
     7,
     8,
     9,
     10,
     11,
     12,
     14,
     15,
     16,
     17,
     18,
     19,
     21,
     22,
     23,
     24,
     25,
     26,
     28,
     29,
     30,
     31,
     32,
     33,
     35,
     36,
     37,
     38,
     39,
     40,
     42,
     43,
     44,
     45,
     46,
     47,
     49,
     50,
     51,
     52,
     53,
     54,
     56,
     57,
     58,
     59,
     60,
     61,
     63,
     64,
     65,
     66,
     67,
     68]



You can also ask for just the frames of a specific z plane

.. code:: ipython3

    im_par.flatten_by_timepoints(timepoint_start = 0, timepoint_end = 10, reference_z = 3)




.. parsed-literal::

    [3, 10, 17, 24, 31, 38, 45, 52, 59, 66]



If you want all of the frames corresponding to a given
slice/color/whatever, use the ``framelist_by_x`` methods:

.. code:: ipython3

    print ("All frames with color channel 0:")
    print(im_par.framelist_by_color(color_channel = 0, lower_bound_timepoint = 0, upper_bound_timepoint=10))
    
    print("All frames in timepoint < 5 in the third slice:")
    print(im_par.framelist_by_slices(color_channel=0, lower_bound = 0, upper_bound=5, slices = [2]))


.. parsed-literal::

    All frames with color channel 0:
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68]
    All frames in timepoint < 5 in the third slice:
    [2, 9, 16, 23, 30]


