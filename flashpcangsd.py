"""
Our FlashPCAngsd approach.
"""

__author__ = "Jonas Meisner"

# Libraries and modules
import numpy as np
import threading
import os
import argparse
from numba import jit
from math import sqrt
from scipy.sparse.linalg import svds

##### Argparse #####
parser = argparse.ArgumentParser(prog="FlashPCAngsd")
parser.add_argument("--version", action="version", version="%(prog)s alpha 0.1")
parser.add_argument("-i", metavar="FILE",
	help="Input file (.npy)")
parser.add_argument("-e", metavar="INT", type=int,
	help="Number of eigenvectors to use")
parser.add_argument("-m", metavar="INT", type=int, default=100,
	help="Maximum iterations for estimation of individual allele frequencies (100)")
parser.add_argument("-m_tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for update in estimation of individual allele frequencies (1e-5)")
parser.add_argument("-t", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-cov_save", action="store_true",
	help="Save estimated covariance matrix (Binary)")
parser.add_argument("-indf_save", action="store_true",
	help="Save estimated allele frequencies (Binary)")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="flash")
args = parser.parse_args()


### Functions ###
# Estimate population allele frequencies
def estimateF(D, t=1):
	n, m = D.shape
	f = np.zeros(m, dtype=np.float32)

	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=estimateF_inner, args=(D, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return f

# Inner function to estimate population allele frequencies
@jit("void(i1[:, :], f4[:], i8, i8)", nopython=True, nogil=True, cache=True)
def estimateF_inner(D, f, S, N):
	n, m = D.shape # Dimensions
	for j in xrange(S, min(S+N, m)):
		nSite = 0
		for i in xrange(n):
			if D[i, j] != -9:
				nSite += 1
				f[j] += D[i, j]
		f[j] /= nSite

# Center dosages prior to SVD - (E - f)
@jit("void(f4[:, :], f4[:], i8, i8)", nopython=True, nogil=True, cache=True)
def centerE(E, f, S, N):
	n, m = E.shape
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			E[i, j] -= f[j]

# Compute individual allele frequencies - add intercept and truncate
@jit("void(f4[:, :], f4[:], i8, i8)", nopython=True, nogil=True, cache=True)
def computePi(Pi, f, S, N):
	n, m = Pi.shape
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			Pi[i, j] += f[j]
			Pi[i, j] = max(Pi[i, j], 1e-4)
			Pi[i, j] = min(Pi[i, j], 1-(1e-4))

# Iteration for estimation of individual allele frequencies
def computeSVD(E, f, e, chunks, chunk_N):
	# Multithreading - Centering dosages
	threads = [threading.Thread(target=centerE, args=(E, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Reduced SVD of rank K (Scipy library)
	W, s, U = svds(E, k=e)
	Pi = np.dot(W*s, U)

	# Multithreading - Estimate Pi
	threads = [threading.Thread(target=computePi, args=(Pi, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return Pi, W

# Update E - initial step
@jit("void(i1[:, :], f4[:], i8, i8, f4[:, :])", nopython=True, nogil=True, cache=True)
def updateE_init(D, f, S, N, E):
	n, m = E.shape # Dimensions
	for i in xrange(S, min(S+N, n)):
		# Estimate posterior probabilities and update dosages
		for j in xrange(m):
			if D[i, j] == -9: # Missing site
				E[i, j] = f[j]
			else:
				E[i, j] = D[i, j]

# Update E
@jit("void(i1[:, :], f4[:, :], i8, i8, f4[:, :])", nopython=True, nogil=True, cache=True)
def updateE(D, Pi, S, N, E):
	n, m = E.shape # Dimensions
	for i in xrange(S, min(S+N, n)):
		# Estimate posterior probabilities and update dosages
		for j in xrange(m):
			if D[i, j] == -9: # Missing site
				E[i, j] = Pi[i, j]
			else:
				E[i, j] = D[i, j]

# Standardize dosages prior to final SVD - (E - f)/sqrt(f*(1-f))
@jit("void(f4[:, :], f4[:], i8, i8)", nopython=True, nogil=True, cache=True)
def standardizeE(E, f, S, N):
	n, m = E.shape
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			E[i, j] -= f[j]
			E[i, j] /= sqrt(f[j]*(1 - f[j]))

# Final SVD for extracting V and Sigma
def finalSVD(E, f, e, chunks, chunk_N):
	n, m = E.shape

	# Multithreading
	threads = [threading.Thread(target=standardizeE, args=(E, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	V, s, U = svds(E, k=e)
	Sigma = s**2/m
	return V[:, ::-1], Sigma[::-1]

# Measure difference
def rmse(A, B, chunks, chunk_N):
	n, m = A.shape
	R = np.zeros(n, dtype=np.float32)

	# Multithreading
	threads = [threading.Thread(target=rmse_inner, args=(A, B, chunk, chunk_N, R)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return sqrt(np.sum(R)/(n*m))

@jit("void(f4[:, :], f4[:, :], i8, i8, f4[:])", nopython=True, nogil=True, cache=True)
def rmse_inner(A, B, S, N, R):
	n, m = A.shape
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			if np.sign(A[i, j]) == np.sign(B[i,j]):
				R[i] += (A[i, j] - B[i, j])*(A[i, j] - B[i, j])
			else:
				C = A[i, j]*-1
				R[i] += (C - B[i, j])*(C - B[i, j])		


### Main function ###
def flashPCAngsd(D, f, e, M=100, M_tole=1e-5, t=1):
	n, m = D.shape # Dimensions
	E = np.empty((n, m), dtype=np.float32) # Initiate E

	# Multithreading parameters
	chunk_N = int(np.ceil(float(n)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=updateE_init, args=(D, f, chunk, chunk_N, E)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	if M < 2:
		print "Missingess not taken into account!"

		# Estimate approximate covariance matrix based on reconstruction
		print "Inferring set of eigenvectors."
		V, Sigma = finalSVD(E, f, e, chunks, chunk_N)

		# Estimate approximate covariance matrix
		print "Approximating covariance matrix.\n"
		C = np.dot(V*Sigma, V.T)
	
		return V, C, None
	else:
		# Estimate initial individual allele frequencies
		Pi, W = computeSVD(E, f, e, chunks, chunk_N)
		prevW = np.copy(W)
		print "Individual allele frequencies estimated (1)"
		
		# Iterative estimation of individual allele frequencies
		for iteration in xrange(2, M+1):
			# Multithreading
			threads = [threading.Thread(target=updateE, args=(D, Pi, chunk, chunk_N, E)) for chunk in chunks]
			for thread in threads:
				thread.start()
			for thread in threads:
				thread.join()

			# Estimate individual allele frequencies
			Pi, W = computeSVD(E, f, e, chunks, chunk_N)

			# Break iterative update if converged
			diff = rmse(W, prevW, chunks, chunk_N)
			print "Individual allele frequencies estimated (" + str(iteration) + "). RMSE=" + str(diff)
			if diff < M_tole:
				print "Estimation of individual allele frequencies has converged."
				break
			prevW = np.copy(W)

		del W, prevW

		# Multithreading
		threads = [threading.Thread(target=updateE, args=(D, Pi, chunk, chunk_N, E)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Estimate approximate covariance matrix based on reconstruction
		print "Inferring final set of eigenvectors."
		V, Sigma = finalSVD(E, f, e, chunks, chunk_N)

		# Estimate approximate covariance matrix
		print "Approximating covariance matrix.\n"
		C = np.dot(V*Sigma, V.T)
	
		return V, C, Pi


### Caller ###
print "FlashPCAngsd Alpha 0.1\n"

# Read in single-read matrix
print "Reading in single-read sampling matrix."
D = np.load(args.i).astype(np.int8, copy=False)
n, m = D.shape

# Multithreading parameters
chunk_N = int(np.ceil(float(n)/args.t))
chunks = [i * chunk_N for i in xrange(args.t)]

# Population allele frequencies
print "Estimating population allele frequencies."
f = estimateF(D, args.t)
mask = (f >= 0.05) & (f <= 0.95)

# Removing rare variants
print "Filtering out rare variants."
f = np.compress(mask, f)
D = np.compress(mask, D, axis=1)
n, m = D.shape
print str(n) + " samples, " + str(m) + " sites.\n"

# FlashPCAngsd
print "Performing FlashPCAngsd."
V, C, Pi = flashPCAngsd(D, f, args.e, args.m, args.m_tole, args.t)

print "Saving eigenvectors as " + args.o + ".eigenvecs.npy (Binary)."
np.save(args.o + ".eigenvecs", V.astype(float, copy=False))

if args.cov_save:
	print "Saving covariance matrix as " + args.o + ".cov.npy (Binary)."
	np.savetxt(args.o + ".cov", C.astype(float, copy=False))

if args.indf_save:
	print "Saving individual allele frequencies as " + args.o + ".pi.npy (Binary)."
	np.save(args.o + ".pi", Pi.astype(float, copy=False))