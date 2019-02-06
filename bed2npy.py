"""
Converting PLINK to binary NumPy array.
"""

__author__ = "Jonas Meisner"

# Libraries and modules
import numpy as np
import threading
import argparse
from numba import jit
from pandas_plink import read_plink
from tqdm import tqdm

##### Argparse #####
parser = argparse.ArgumentParser(prog="bed2npy")
parser.add_argument("-plink", metavar="PREFIX",
	help="Prefix for binary PLINK files")
parser.add_argument("-chunks", metavar="INT", type=int, default=50,
	help="Number of chunks used for reading PLINK bed file (50)")
parser.add_argument("-t", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-o", metavar="OUTPUT", help="Output filename", default="flash")
args = parser.parse_args()


### Functions ###
def convertPlink(G_chunk, t=1):
	n, m = G_chunk.shape # Dimensions
	D_chunk = np.empty((n, m), dtype=np.int8)

	# Multithreading parameters
	chunk_N = int(np.ceil(float(n)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=convertPlink_inner, args=(G_chunk, D_chunk, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return D_chunk

# Inner function for converting PLINK files to D matrix
@jit("void(f8[:, :], i1[:, :], i8, i8)", nopython=True, nogil=True, cache=True)
def convertPlink_inner(G, D, S, N):
	n, m = D.shape # Dimensions
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			if np.isnan(G[i, j]): # Missing value
				D[i, j] = -9
			elif int(G[i, j]) == 2:
				D[i, j] = 0
			elif int(G[i, j]) == 0:
				D[i, j] = 1
			else:
				raise ValueError("Check genotype input structure!")


### Read from binary PLINK files ###
print "Reading PLINK files"
_, _, G = read_plink(args.plink)
m, n = G.shape

# Construct single-read matrix from PLINK files
print "Converting to single-read sampling matrix."
D = np.empty((n, m), dtype=np.int8)

# Loop over chunks of bed file
idx_split = np.array_split(np.arange(m, dtype=int), args.chunks)
for c in tqdm(range(args.chunks), desc="Reading chunks"):
	G_chunk = G[idx_split[c], :].compute().T
	D[:, idx_split[c]] = convertPlink(G_chunk, args.t)
	del G, G_chunk

# Saving 8-bit integer matrix
np.save(args.o, D)