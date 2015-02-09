from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys, os

setup(
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(),],
)

