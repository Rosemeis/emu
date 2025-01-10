from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		name="emu.shared",
		sources=["emu/shared.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), 
	Extension(
		name="emu.memory",
		sources=["emu/memory.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="emu-popgen",
	version="1.2.0",
	author="Jonas Meisner",
	author_email="meisnerucph@gmail.com",
	description="EM-PCA for inferring population structure in the presence of missingness",
	long_description_content_type="text/markdown",
	long_description=open("README.md").read(),
	url="https://github.com/Rosemeis/emu",
	classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
	ext_modules=cythonize(extensions),
	python_requires=">=3.10",
	install_requires=[
		"cython>3.0.0",
		"numpy>2.0.0"
	],
	packages=["emu"],
	entry_points={
		"console_scripts": ["emu=emu.main:main"]
	},
)
