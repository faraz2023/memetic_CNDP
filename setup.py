#python setup.py build_ext --inplace



from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
setup(
    cmdclass = {'build_ext':build_ext},

    #################for ubuntu compile
    ext_modules = cythonize([
                    Extension('primes', sources = ['primecounter.pyx'], include_dirs=[np.get_include()]),
                   ], annotate=True)
)