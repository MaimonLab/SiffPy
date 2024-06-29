from setuptools import setup, Extension
import platform
import sys
import subprocess
      
try:
   import numpy
except ImportError:
   raise Exception("Numpy is not yet installed on this distribution. Install numpy with pip or conda")

from setuptools.command.install import install

#from setuptools.command.install import clean

define_macros = None
extra_compile_args = None
library_dirs = None
libraries = None


# CHANGE THIS TO TRUE FOR DEBUG MODE
# or maybe I should make this a setting in the package itself?
# TODO: debug as a command line option
#DEBUG = True
DEBUG = False

class MSVCInstallCommand(install):             
   """
   To try to install MSVC build tools on Windows
   """
   def has_msvc_build_tools(self)->bool:
      # Check if cl.exe is available in PATH
      try:
         subprocess.check_call(
            ['cl'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
         )
         return True
      except Exception:
         return False
      return False

   def install_msvc_build_tools(self):
      try:
         subprocess.check_call(
            [
               'choco',
               'install',
               'visualstudio2019buildtools',
               '-y',
            ]
         )
      except Exception:
         print(
            "Could not install MSVC Build Tools with Chocolatey."
            + " Please install them manually with"
            + "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
         )

   def run(self):
      if not self.has_msvc_build_tools():
         print("MSVC Build Tools not found." 
               + " Attempting to install with Chocolatey..."
         )
         self.install_msvc_build_tools()
      super().run()


installclass = install
if platform.system() == 'Windows':
   installclass = MSVCInstallCommand

if DEBUG:
   define_macros = [('__DEBUG', 1)]

if platform.system() == 'Windows':
   extra_compile_args = ["/std:c++17"]
else:
   extra_compile_args = [
      "-std=c++11",
      #"-Werror"
   ]

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

def setupcall():
   setup (
      packages = ['siffpy'],
      ext_modules = [siffmodule],
      cmdclass={
         'install': installclass,
      },
      python_requires='>=3.9.0',
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