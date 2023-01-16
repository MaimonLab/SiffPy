from setuptools import setup, Extension
import setuptools
import glob

try:
    import numpy
except ImportError as error:
    raise Exception("Numpy is not yet installed on this distribution. Set up numpy using command 'pip install <directory_containing_this_setup.py>' instead.")

# written for Darwin. Probably should write a Windows and/or GNU compatible
siffmodule = Extension(
                    name='siffpy.siffreadermodule',
                    sources = glob.glob('./src/*.cpp'),
                    include_dirs = [
                        numpy.get_include()
                    ]
                    ,
                    extra_compile_args=["-stdlib=libc++"]
                    ,
                    language="c++"
                    )

setup (
    packages=['siffpy.siffreadermodule'],
    ext_modules = [siffmodule],
)