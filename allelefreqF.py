"""
Generating initial allele frequency guess.
"""

__author__ = "Jonas Meisner"

# Libraries and modules
import numpy as np
import threading
import argparse
from numba import jit

##### Argparse #####
parser = argparse.ArgumentParser(prog="bed2npy")
parser.add_argument("-npy", metavar="PREFIX",
	help="Input file (.npy)")
parser.add_argument("-reg", metavar="FILE",
	help="Region index vector for individuals (0-indexed)")
parser.add_argument("-t", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-o", metavar="OUTPUT", help="Output filename", default="flash")
args = parser.parse_args()


### Functions ###
# Estimate population allele frequencies
def estimateF_multi(D, Pvec, t=1):
	n, m = D.shape # Dimensions
	K = max(Pvec) + 1
	F = np.zeros((m, K), dtype=np.float32)	

	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=estimateF_multi_inner, args=(D, F, Pvec, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return F

# Inner function to estimate population allele frequencies
@jit("void(i1[:, :], f4[:, :], i4[:], i8, i8)", nopython=True, nogil=True, cache=True)
def estimateF_multi_inner(D, F, Pvec, S, N):
	n, m = D.shape # Dimensions
	K = F.shape[1]
	for j in xrange(S, min(S+N, m)):
		nSite = np.zeros(K, dtype=np.int32)
		for i in xrange(n):
			for k in xrange(K):
				if Pvec[i] == k:
					if D[i, j] != -9:
						nSite[k] += 1
						F[j, k] += D[i, j]
		
		for k in xrange(K):
			if nSite[k] == 0:
				F[j, k] = 0
			else:
				F[j, k] /= nSite[k]


### Running script
print "Reading in single-read sampling matrix from binary NumPy file."
# Read from binary NumPy file. Expects np.int8 data format
D = np.load(args.npy)
assert D.dtype == np.int8, "NumPy array must be of 8-bit integer format (np.int8)!"

# Read index vector for indivudals
Pvec = np.genfromtxt(args.reg, dtype=np.int32)

# Estimate allele frequencies
F = estimateF_multi(D, Pvec, args.t)

# Save region-based allele frequencies
np.save(args.o, F)
