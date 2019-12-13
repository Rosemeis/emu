"""
Converts matrix format into Numpy binary format for EMU.
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
parser.add_argument("-plink", metavar="PREFIX",
	help="Prefix for binary PLINK files (.bed, .bim, .fam)")
parser.add_argument("-block", action="store_true",
	help="Convert PLINK file in blocks")
parser.add_argument("-block_size", metavar="INT", type=int, default=1024,
	help="Block size used")
parser.add_argument("-min_missing_rate", metavar="FLOAT", type=float,
	help="Set a minimum rate for missingness per site")
parser.add_argument("-max_missing_rate", metavar="FLOAT", type=float,
	help="Set a maximum rate for missingness per site")
parser.add_argument("-old_mat", metavar="FILE",
	help="Gzipped input file (.mat.gz) - (LEGACY)")
parser.add_argument("-depth", metavar="INT", type=int, default=0,
	help="Specify the threshold of depth for each site - (LEGACY)")
parser.add_argument("-ped", metavar="FILE",
	help="Info from individuals (.ped), only for '-old_mat'")
parser.add_argument("-binary", action="store_true",
	help="Output data in Binary file - 8-bit signed chars")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="matrix")
args = parser.parse_args()

### Main ###
if args.plink is not None:
	print("Parsing PLINK files")
	from pandas_plink import read_plink
	bim, fam, bed = read_plink(args.plink, verbose=False)
	bed = bed.astype(np.float32)
	m, n = bed.shape
	print("Matrix size: (" + str(n) + " x " + str(m) + ")")
	
	if not args.block:
		D = np.empty((n, m), dtype=np.int8)
		reader.convertBed(D, bed.T.compute())
		if args.binary:
			print("Saving matrix as binary file - 8-bit signed chars")
			bfile = open(args.o + ".bin", "wb")
			D.tofile(bfile)
			bfile.close()
		else:
			print("Saving matrix in binary Numpy format (.npy)")
			np.save(args.o, D)
	else:
		assert args.binary == True, "Conversion in blocks must be in binary output only!"
		D = np.empty((m, n), dtype=np.int8)
		with open(args.o + ".bin", "ab") as bfile:
			for b in range(0, m, args.block_size):
				if (b + args.block_size) > m:
					snpBlock = bed[b:].compute()
					reader.convertBed(D[b:], snpBlock)
				else:
					reader.convertBed(D[b:(b + args.block_size)], bed[b:(b + args.block_size)].compute())
			D.T.tofile(bfile)


elif args.mat is not None:
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