from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils import log as distutils_logger
from distutils.errors import DistutilsSetupError
from Cython.Build import cythonize
import os, subprocess
import glob
import numpy


ext_modules = [
    Extension("pyzfp",
              sources=["pyzfp.pyx"],
              include_dirs=['zfp-0.5.3/include',
                            numpy.get_include()],
              libraries=["zfp"],  # Unix-like specific,
              library_dirs=["zfp-0.5.3/lib"],
              extra_compile_args=['-fopenmp'],
             extra_link_args=['-fopenmp', '-Wl,-rpath,/usr/local/lib']
              )
]

class specialized_build_ext(build_ext, object):
    """
    Specialized builder for testlib library

    """
    special_extension = ext_modules[0].name

    def build_extension(self, ext):

        if ext.name!=self.special_extension:
            # Handle unspecial extensions with the parent class' method
            super(specialized_build_ext, self).build_extension(ext)
        else:
            # Handle special extension
            sources = ext.sources
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                       "in 'ext_modules' option (extension '%s'), "
                       "'sources' must be present and must be "
                       "a list of source filenames" % ext.name)
            sources = list(sources)

            if len(sources)>1:
                sources_path = os.path.commonpath(sources)
            else:
                sources_path = os.path.dirname(sources[0])
            sources_path = os.path.realpath(sources_path)
            if not sources_path.endswith(os.path.sep):
                sources_path+= os.path.sep

            if not os.path.exists(sources_path) or not os.path.isdir(sources_path):
                raise DistutilsSetupError(
                       "in 'extensions' option (extension '%s'), "
                       "the supplied 'sources' base dir "
                       "must exist" % ext.name)

            output_dir = os.path.realpath(os.path.join(sources_path,'..','lib'))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_lib = 'libtestlib.a'

            command = 'make'
            distutils_logger.info('Will execute the following command in with subprocess.Popen: \n{0}'.format(command))


            make_process = subprocess.Popen(command,
                                            cwd=sources_path,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            shell=True)
            stdout, stderr = make_process.communicate()
            distutils_logger.debug(stdout)
            #if stderr:
            #    raise DistutilsSetupError('An ERROR occured while running the '
            #                              'Makefile for the {0} library. '
            #                              'Error status: {1}'.format(output_lib, stderr))
            # After making the library build the c library's python interface with the parent build_extension method
            super(specialized_build_ext, self).build_extension(ext)


setup(name="pyzfp",
      ext_modules=cythonize(ext_modules, gdb_debug=True), 
      cmdclass={'build_ext': specialized_build_ext})
