from setuptools import setup, Extension
import setuptools
import logging
import os, re

try:
    import numpy
    import scipy
except ImportError as error:
    raise Exception("Numpy or scipy is not yet installed on this distribution. Set up numpy using command 'pip install <directory_containing_this_setup.py>' instead.")

HERE = os.path.abspath(os.path.dirname(__file__))

def _version() -> str:
    """ Parses _version.py to return a version string without executing the code """
    with open(os.path.join(HERE, "siffpy", "core", "_version.py")) as f:
        match = re.search(r'version\s?=\s?\'([^\']+)', f.read())
        if match:
            return match.groups()[0].split('+')[0]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# written for Darwin. Probably should write a Windows and/or GNU compatible
siffmodule = Extension('siffreader',
                    sources = [
                        'siffpy/siffreadermodule/src/siffreadermodule.cpp', 
                        'siffpy/siffreadermodule/src/siffreader.cpp',
                        'siffpy/siffreadermodule/src/sifftotiff.cpp'
                    ],
                    include_dirs = [
                        'siffpy/siffreadermodule/include',
                        numpy.get_include()
                    ]
                    ,
                    extra_compile_args=["-stdlib=libc++"]
                    ,
                    language="c++"
                    )

setup (name = 'siffpy',
       version = _version(),
       install_requires = [
           'numpy',
           'scipy'
       ],
       setup_requires = [
           'numpy',
           'scipy'
       ],
       extras_require = {
           'viz' : ['holoviews','bokeh','napari']
       },
       description = 'Python package for reading and processing .siffs and ScanImage .tiffs',
       ext_modules = [siffmodule],
       packages = setuptools.find_packages(),
       license='GPL-3',
       classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        ],
        author_email='thornquist@rockefeller.edu',
        author='Stephen Thornquist'
       )
    
try:
    import holoviews
    import bokeh
except ImportError as error:
    logging.warning(f"{bcolors.WARNING}\nWARNING:\n\tInstalled without HoloViews or Bokeh. Plotting functionality may fail until those are installed.{bcolors.ENDC}")

try:
    import napari
except ImportError as error:
    logging.warning(f"{bcolors.WARNING}\nWARNING:\n\tInstalled without napari." +
    " All functionality should be present, but napari can provide a more pleasant" + 
    "experience for interacting with image data. Try to install it (ideally with pip)"+
    f" using the command 'pip install napari'. Has a tendency to break with conda.{bcolors.ENDC}")

try:
    import dask
except ImportError as error:
    logging.warning(f"{bcolors.WARNING}\nWARNING:\n\tInstalled without dask." +
    " Some napari plotting functionality will fail (anything relying on delayed " +
    f"evaluation. Install with conda or pip.){bcolors.ENDC}")