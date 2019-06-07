from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension(
				"reader",
				["reader.pyx"],
				include_dirs=[numpy.get_include()],
				language="c++"
			),
			Extension(
				"shared",
				["shared.pyx"],
				extra_compile_args=['-fopenmp'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()],
			),
			Extension(
				"halko",
				["halko.pyx"],
				extra_compile_args=['-fopenmp'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()],
			)]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)