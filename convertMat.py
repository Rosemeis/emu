"""
Converts matrix format into Numpy binary format for FlashPCAngsd.
Can also generate index file that is used for guided estimation based on ped-file.

Example usage: python convertMat.py matrix.mat.gz -o flash
"""

__author__ = "Jonas Meisner"

# Libraries
import reader
import numpy as np
import argparse
import os

##### Argparse #####
parser = argparse.ArgumentParser()
parser.add_argument("mat", metavar="FILE",
	help="Gzipped input file (.mat.gz)")
parser.add_argument("-depth", metavar="INT", type=int, default=0,
	help="Specify the threshold of depth for each site")
parser.add_argument("-ped", metavar="FILE",
	help="Info from individuals (.ped)")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="matrix")
args = parser.parse_args()

### Main ###
print("Reading matrix.")
D = reader.readMat(args.mat, args.depth).T
n, m = D.shape
print("Matrix size: (" + str(n) + " x " + str(m) + ")")
print("Saving matrix in binary Numpy format (.npy)")
np.save(args.o, D)
del D

if args.ped is not None:
	print("Generating index vector for individuals in guided estimation.")
	import pandas as pd
	f = pd.read_csv(args.ped, sep="\t", usecols=[2], squeeze=True)
	U = f.unique()
	K = U.shape[0]
	p = np.zeros(f.shape[0], dtype=np.int8)

	for k in range(K):
		mask = f == U[k]
		p[mask.values] = k

	print("Saving index vector in binary Numpy format (.npy)")
	np.save(args.o + ".index", p)