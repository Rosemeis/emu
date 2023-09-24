from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"src.shared_cy",
		["src/shared_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), Extension(
		"src.memory_cy",
		["src/memory_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="emu",
	version="1.0",
	description="EM-PCA for performing PCA in the presence of missingness",
	author="Jonas Meisner",
	packages=["src"],
	entry_points={
		"console_scripts": ["emu=src.main:main"]
	},
	python_requires=">=3.6",
	install_requires=[
		"cython",
		"numpy"
	],
	ext_modules=cythonize(extensions),
	include_dirs=[numpy.get_include()]
)
