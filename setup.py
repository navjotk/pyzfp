from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

ext_modules = [
    Extension("pyzfp",
              sources=["zfp.pyx"],
              include_dirs=['zfp-0.5.3/include',
                            numpy.get_include()],
              libraries=["zfp"]  # Unix-like specific
              )
]

setup(name="pyzfp",
      ext_modules=cythonize(ext_modules))
