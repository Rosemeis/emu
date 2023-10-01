from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"emu.shared_cy",
		["emu/shared_cy.pyx"],
		extra_compile_args=['-fopenmp', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), 
	Extension(
		"emu.memory_cy",
		["emu/memory_cy.pyx"],
		extra_compile_args=['-fopenmp', '-g0', '-Wno-unreachable-code'],
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
	packages=["emu"],
	entry_points={
		"console_scripts": ["emu=emu.main:main"]
	},
	python_requires=">=3.6",
	install_requires=[
		"cython",
		"numpy"
	],
	ext_modules=cythonize(extensions),
	include_dirs=[numpy.get_include()]
)
