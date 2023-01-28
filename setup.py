from setuptools import setup, Extension

try:
   import numpy
except ImportError as error:
   raise Exception("Numpy is not yet installed on this distribution. Set up numpy using command 'pip install <directory_containing_this_setup.py>' instead.")

#written for Darwin. Probably should write a Windows and/or GNU compatible
siffmodule = Extension(
   name='siffreadermodule',
   sources = [
       'siffreadermodule/src/siffio.cpp',
       'siffreadermodule/src/siffreader.cpp',
       'siffreadermodule/src/siffreadermodule.cpp',
       'siffreadermodule/src/sifftotiff.cpp',
   ],
   include_dirs = [
       numpy.get_include()
   ],
   extra_compile_args=["-stdlib=libc++"],
   language="c++",
)

setup (
   packages = ['siffpy'],
   ext_modules = [siffmodule],
)