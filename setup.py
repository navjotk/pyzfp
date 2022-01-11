from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils import log as distutils_logger
from distutils.errors import DistutilsSetupError

import os
import subprocess
import setuptools

ZFP_DOWNLOAD_PATH = 'https://github.com/LLNL/zfp/releases/download/0.5.5/zfp-0.5.5.tar.gz'  # noqa


def download_file(url):
    import requests
    fname = url.split("/")[-1]
    r = requests.get(url)
    with open(fname, 'wb') as f:
        f.write(r.content)


def has_flag(compiler, flagname):
    """Check whether a flag is supported on a compiler."""
    import tempfile
    from distutils.errors import CompileError
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except CompileError:
            return False
    return True


def flag_filter(compiler, *flags):
    """Filter flags, returns list of accepted flags"""
    result = []
    for flag in flags:
        if has_flag(compiler, flag):
            result.append(flag)
    return result


class lazy_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():
    import numpy
    from Cython.Build import cythonize
    ext = Extension("pyzfp",
                    sources=["pyzfp.pyx"],
                    include_dirs=['zfp-0.5.5/include',
                                  numpy.get_include()],
                    libraries=["zfp"],  # Unix-like specific,
                    library_dirs=["zfp-0.5.5/lib"],
                    # extra_link_args=['-Wl,-rpath,/usr/local/lib']
                    )
    return cythonize([ext])


class specialized_build_ext(build_ext, object):
    """
    Specialized builder for testlib library
    Code borrowed from: https://stackoverflow.com/a/48641638

    """
    special_extension = "pyzfp"

    def build_extension(self, ext):
        if has_flag(self.compiler, '-fopenmp'):
            for ext in self.extensions:
                ext.extra_compile_args += ['-fopenmp']
                ext.extra_link_args += ['-fopenmp']
            clang = False
        else:
            clang = True

        if ext.name != self.special_extension:
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
            if len(sources) > 1:
                sources_path = os.path.commonpath(sources)
            else:
                sources_path = os.path.dirname(sources[0])
            sources_path = os.path.realpath(sources_path)
            if not sources_path.endswith(os.path.sep):
                sources_path += os.path.sep

            if not os.path.exists(sources_path) or \
               not os.path.isdir(sources_path):
                raise DistutilsSetupError(
                       "in 'extensions' option (extension '%s'), "
                       "the supplied 'sources' base dir "
                       "must exist" % ext.name)

            download_file(ZFP_DOWNLOAD_PATH)
            command = 'make'
            if clang:
                command += ' OPENMP=0'
            else:
                command += ' OPENMP=1'

            env_vars = ['CC', 'CXX', 'CFLAGS', 'FC']

            for v in env_vars:
                val = os.getenv(v)
                if val is not None:
                    command += ' %s=%s' % (v, val)

            distutils_logger.info('Will execute the following command in ' +
                                  'with subprocess.Popen:' +
                                  '\n{0}'.format(command))
            try:
                output = subprocess.check_output(command,
                                                 cwd=sources_path,
                                                 stderr=subprocess.STDOUT,
                                                 shell=True)
            except subprocess.CalledProcessError as e:
                distutils_logger.info(str(e.output))
                raise

            distutils_logger.info(str(output))

            # After making the library build the c library's python interface
            # with the parent build_extension method
            super(specialized_build_ext, self).build_extension(ext)


with open("README.md", "r") as fh:
    long_description = fh.read()


configuration = {
    'name': 'pyzfp',
    'packages': setuptools.find_packages(),
    'setup_requires': ['cython>=0.17', 'requests', 'numpy', 'setuptools_scm'],
    'ext_modules': lazy_cythonize(extensions),
    'use_scm_version': True,
    'cmdclass': {'build_ext': specialized_build_ext},
    'description': "A python wrapper for the ZFP compression libary",
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'url': 'https://github.com/navjotk/pyzfp',
    'author': "Navjot Kukreja",
    'author_email': 'navjotk@gmail.com',
    'license': 'MIT',
}


setup(**configuration)
