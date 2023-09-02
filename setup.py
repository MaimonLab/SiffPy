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
   # For some reason libstdc++ doesn't define <string>?
   # Sadly libc++ has painfully slow regex, but for now
   # I don't need to rely on regex tools.
   extra_compile_args = ["-stdlib=libc++"]
else:
   print(
      """
      SiffPy's `siffreadermodule` has only been tested on
      MacOS with a python distribution compiled with clang.
      Your Python was apparently not built with clang,
      or you are not on MacOS!
      No guarantees it will work for you!

      That said, all C++ code is built with the
      standard C++11 tools, so it should work on
      any platform with a C++11 compiler 
      """
   )

# TDOO: decide if I want to 
#written for Darwin. Probably should write a Windows and/or GNU compatible
siffmodule = Extension(
   name='siffreadermodule',
   sources = [
       'siffreadermodule/src/siffio.cpp',
       'siffreadermodule/src/siffreader/siffreader.cpp',
       'siffreadermodule/src/siffreader/flim/histogram_methods.cpp',
       'siffreadermodule/src/siffreader/flim/lifetime_methods.cpp',
       'siffreadermodule/src/siffreader/intensity/intensity_methods.cpp',
       'siffreadermodule/src/siffreader/time/time_methods.cpp',
       'siffreadermodule/src/siffreadermodule.cpp',
       'siffreadermodule/src/framedata.cpp',
       #'siffreadermodule/src/sifftotiff.cpp',
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