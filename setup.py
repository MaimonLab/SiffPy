from setuptools import setup, Extension

try:
   import numpy
except ImportError as error:
   raise Exception("Numpy is not yet installed on this distribution. Install numpy with pip or conda")

import platform, sys

from setuptools.command.install import install
#from setuptools.command.install import clean

define_macros = None
extra_compile_args = None
library_dirs = None
libraries = None


# CHANGE THIS TO TRUE FOR DEBUG MODE
# TODO: debug as a command line option
DEBUG = True

# Undoing this one for now.... seems like options in pip are being deprecated

# class CleanInstallCommand(install):             
#    """ Provides the '--config-settings=debug' option to setup.py install """
#    user_options = install.user_options + [
#       ('debug', None, 'Installs in debug mode (logs operations)'),
#    ]                                      

#    def initialize_options(self):          
#       self.debug = None
#       super().initialize_options()   

#    def finalize_options(self):
#       if self.debug:
#          self.debug = True
#          assert False
#       else:
#          self.debug = False
#       super().finalize_options()            

#    def run(self):
#       global DEBUG
#       DEBUG = self.debug
#       print("DEBUG MODE: ", DEBUG) 
#       super().run()  
#       c = clean(self.distribution)
#       c.all = True
#       c.finalize_options()
#       c.run()     

installclass = install
#installclass = CleanInstallCommand

if DEBUG:
   define_macros = [('__DEBUG', 1)]

if platform.system() == 'Windows':
   extra_compile_args = ["/std:c++17"]
else:
   extra_compile_args = [
      "-std=c++11",
      #"-Werror"
   ]

if not (
      (platform.system() == 'Darwin') and
      ('Clang' in sys.version)
   ):
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

siffmodule = Extension(
   name='siffreadermodule',
   sources = [
       #'siffreadermodule/src/siffio.cpp',
       'siffreadermodule/src/siffreader/siffreader.cpp',
       'siffreadermodule/src/siffreader/siffreader_writer.cpp',
       'siffreadermodule/src/siffreader/flim/histogram_methods.cpp',
       'siffreadermodule/src/siffreader/flim/lifetime_methods.cpp',
       'siffreadermodule/src/siffreader/intensity/intensity_methods.cpp',
       'siffreadermodule/src/siffreader/time/time_methods.cpp',
       'siffreadermodule/src/siffreadermodule.cpp',
       'siffreadermodule/src/framedata.cpp',
       'siffreadermodule/src/sifftotiff.cpp',
       #'siffreadermodule/src/pyFrameData.cpp',
   ],
   library_dirs=library_dirs,
   include_dirs = [
       numpy.get_include(),
   ],
   libraries = libraries,
   extra_compile_args=extra_compile_args,
   language="c++",
   define_macros = define_macros,
   #use_scm_version=True,
)

# defined once here so that when it gets called
# multiple times I don't have to remember to re-write
# and update bits

setupcall = lambda: setup (
   packages = ['siffpy'],
   ext_modules = [siffmodule],
   cmdclass={
      'install': installclass,
   }
)

try:
   setupcall()
except Exception:
   if (
      (platform.system() == 'Darwin') and
      ('Clang' in sys.version)
   ):
      # For some reason libstdc++ in some compilers doesn't define <string>?
      # Sadly libc++ has painfully slow regex, but now
      # I don't need to rely on regex tools.
      import warnings
      warnings.warn(
         """
         Some Xcode sets don't define <string> in libstdc++,
         so trying again with libc++
         """
      )

      siffmodule.extra_compile_args.append("-stdlib=libc++")
      setupcall()