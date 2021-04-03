from setuptools import setup, Extension
import setuptools

try:
    import numpy
except ImportError as error:
    print("Numpy is not yet installed on this distribution. Set up using command 'pip install .' instead.")
    

# written for Darwin. Probably should write a Windows and/or GNU compatible
siffmodule = Extension('siffreader',
                    sources = ['SiffReader/src/siffreadermodule.cpp', 'SiffReader/src/siffreader.cpp'],
                    include_dirs = [
                        'SiffReader/include⁩',
                        numpy.get_include()
                        ]
                    ,
                    extra_compile_args=["-stdlib=libc++"]
                    ,
                    language="c++"
                    )

setup (name = 'siffpy',
       version = '0.2.01',
       install_requires = [
           'numpy'
       ],
       setup_requires = [
           'numpy'
       ],
       description = 'Python package for reading and processing .siffs and ScanImage .tiffs',
       ext_modules = [siffmodule],
       packages = setuptools.find_packages()
       )