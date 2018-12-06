"""
Converting PLINK to binary NumPy array.
"""

__author__ = "Jonas Meisner"

# Libraries and modules
import numpy as np
import threading
import argparse
from numba import jit
from pysnptools.snpreader import Bed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### Argparse #####
parser = argparse.ArgumentParser(prog="bed2npy")
parser.add_argument("-plink", metavar="PREFIX",
	help="Prefix for binary PLINK files")
parser.add_argument("-t", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-o", metavar="OUTPUT", help="Output filename", default="flash")
args = parser.parse_args()


### Functions ###
def convertPlink(G, t=1):
	n, m = G.shape # Dimensions
	D = np.empty((n, m), dtype=np.int8) # Container for single-read matrix

	# Multithreading parameters
	chunk_N = int(np.ceil(float(n)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=convertPlink_inner, args=(G, D, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return D

# Inner function for converting PLINK files to D matrix
@jit("void(f4[:, :], i1[:, :], i8, i8)", nopython=True, nogil=True, cache=True)
def convertPlink_inner(G, D, S, N):
	n, m = D.shape # Dimensions
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			if np.isnan(G[i, j]): # Missing value
				D[i, j] = -9
			elif int(G[i, j]) == 2:
				D[i, j] = 1
			else:
				D[i, j] = int(G[i, j])


# Read from binary PLINK files
print "Reading PLINK files"
readPlink = Bed(args.plink, count_A1=True).read(dtype=np.float32).val
n, m = readPlink.shape

# Construct single-read matrix from PLINK files
print "Converting to single-read sampling matrix."
D = convertPlink(readPlink, args.t)
del readPlink

# Saving 8-bit integer matrix
np.save(args.o, D)