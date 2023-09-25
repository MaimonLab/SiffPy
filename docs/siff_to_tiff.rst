Siff-to-Tiff functions
======================

You might be using a ``ScanImage-FLIM`` microscope, but only want to use regular
intensity imaging tools and analyses. ``SiffPy`` supports most of what you likely
want to do, or returns a ``numpy`` array that you can just pipe in to some other pathway
or analysis code, but some tools just want a plain ``.tiff``. ``SiffPy`` provides a few
simple functions to convert to ``.tiff``. For now, it only implements the ``ScanImage``
``.tiff`` specification (which is not OME compliant!!), but soon I will add ``OME-TIFF``
support as a keyword argument.


Warning: the function below is autodocumented using the DOCSTRING attached to the
``siffreadermodule``'s ``siff_to_tiff`` function, which may be out of date from the
current implementation. The more reliable documentation is in the ``siff_to_tiff``
function's stub file in ``siffreadermodule/__init__.pyi``.
.. autofunction:: siffreadermodule.siff_to_tiff