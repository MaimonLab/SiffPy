from setuptools import setup, Extension

try:
   import numpy
except ImportError as error:
   raise Exception("Numpy is not yet installed on this distribution. Set up numpy using command 'pip install <directory_containing_this_setup.py>' instead.")

import platform, sys

extra_compile_args = []
if (
      (platform.system() == 'Darwin') and
      ('Clang' in sys.version)
   ):
   extra_compile_args = ["-stdlib=libc++"]
else:
   print("""
      SiffPy's `siffreadermodule` has only been tested on
      MacOS with a python distribution compiled with clang.
      Your Python was apparently not built with clang.
      No guarantees it will work for you!"""
   )


#written for Darwin. Probably should write a Windows and/or GNU compatible
siffmodule = Extension(
   name='siffreadermodule',
   sources = [
       'siffreadermodule/src/siffio.cpp',
       'siffreadermodule/src/siffreader.cpp',
       'siffreadermodule/src/siffreadermodule.cpp',
       'siffreadermodule/src/sifftotiff.cpp',
       #'siffreadermodule/src/pyFrameData.cpp',
   ],
   include_dirs = [
       numpy.get_include()
   ],
   extra_compile_args=extra_compile_args,
   language="c++",
)

setup (
   packages = ['siffpy'],
   ext_modules = [siffmodule],
)