from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os
import glob
import numpy

os.environ["CC"] = "gcc"
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

setup(name="pyzfp",
      ext_modules=cythonize(ext_modules, gdb_debug=True))
