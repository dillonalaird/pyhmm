from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np


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

gaussian_niw_src = ['internals/gaussian_niw_interface.pyx']

setup(name='gaussian_niw',
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('gaussian_niw', gaussian_niw_src,
              language='c++',
              include_dirs=[np.get_include(),],
              extra_compile_args=['-std=c++11'])
      ])
