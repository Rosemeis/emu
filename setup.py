from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension(
				"emu.shared_cy",
				["emu/shared_cy.pyx"],
				extra_compile_args=['-fopenmp', '-g0'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()],
			),
			Extension(
				"emu.halko",
				["emu/halko.pyx"],
				extra_compile_args=['-fopenmp', '-g0'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()],
			)]

setup(
	name="EMU",
	version="0.8",
	description="EM-PCA for performing PCA in the presence of missingness",
	author="Jonas Meisner",
	packages=["emu"],
	entry_points={
		"console_scripts": ["emu=emu.emu:main"]
	},
	python_requires=">=3.6",
	ext_modules=cythonize(extensions),
	include_dirs=[numpy.get_include()]
)
