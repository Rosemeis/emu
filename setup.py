from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension(
				"reader",
				["reader.pyx"],
				extra_compile_args=['-g0'],
				include_dirs=[numpy.get_include()],
				language="c++"
			),
			Extension(
				"shared",
				["shared.pyx"],
				extra_compile_args=['-fopenmp', '-g0'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()],
			),
			Extension(
				"halko",
				["halko.pyx"],
				extra_compile_args=['-fopenmp', '-g0'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()],
			)]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)