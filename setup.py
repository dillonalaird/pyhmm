from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import sys, os


forward_backward_src = ['internals/forward_backward_msgs_interface.pyx']

setup(name='forward_backward',
      cmdclass={'build_ext': build_ext},
      ext_modules=[
        Extension('forward_backward', forward_backward_src,
          language='c++',
          include_dirs=[np.get_include(),],
          extra_compile_args=['-std=c++11'])
      ])

gmm_src = ['internals/gmm_interface.pyx']

setup(name='gmm',
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('gmm', gmm_src,
            language='c++',
            include_dirs=[np.get_include(),],
            extra_compile_args=['-std=c++11'])
      ])
