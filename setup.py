from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import sys, os


py_forward_backward_src = ['internals/forward_backward_msgs.cpp',
                           'internals/forward_backward_msgs_interface.pyx']

setup(name='py_forward_backward',
      cmdclass={'build_ext': build_ext},
      ext_modules=[
        Extension('py_forward_backward', py_forward_backward_src,
          language='c++',
          include_dirs=[np.get_include(),],
          extra_compile_args=['-std=c++11'])
      ])
