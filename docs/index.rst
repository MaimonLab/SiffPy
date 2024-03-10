.. SiffPy documentation master file, created by
   sphinx-quickstart on Sat Sep 16 01:43:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SiffPy's documentation!
==================================

This ``readthedocs`` documentation contains information on:

- how to install SiffPy
- how to use SiffPy to generate arrays of data from ``.siff`` files
- analyses on those data using the ``trace_analysis`` tools in ``siffmath``
- How the ``C++`` ``siffreadermodule`` extension module works

These tools are used to read and manipulate data in ``.siff`` files (ScanImage-FLIM format).

.. toctree::
   :maxdepth: 2
   :caption: Installation:
   
   installation_guide
   additional_dependencies

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   basic_use_notebook
   trace_notebook
   flim_analysis
   siff_to_tiff

.. toctree::
   :maxdepth: 2
   :caption: Main classes:

   siffreader_api
   im_params_api
   flim_params
   trace_analysis_api
   siffreadermodule_api
   registration_info


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
