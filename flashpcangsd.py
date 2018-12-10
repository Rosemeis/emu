"""
Our FlashPCAngsd approach.
Jonas Meisner, Siyang Liu and Anders Albrechtsen (2018)
"""

__author__ = "Jonas Meisner"

# Libraries and modules
import numpy as np
import threading
import argparse
from numba import jit
from math import sqrt
from scipy.sparse.linalg import svds

##### Argparse #####
parser = argparse.ArgumentParser(prog="FlashPCAngsd")
parser.add_argument("--version", action="version", version="%(prog)s alpha 0.175")
parser.add_argument("-npy", metavar="FILE",
	help="Input file (.npy)")
parser.add_argument("-plink", metavar="PREFIX",
	help="Prefix for binary PLINK files")
parser.add_argument("-e", metavar="INT", type=int,
	help="Number of eigenvectors to use")
parser.add_argument("-m", metavar="INT", type=int, default=100,
	help="Maximum iterations for estimation of individual allele frequencies (100)")
parser.add_argument("-m_tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for update in estimation of individual allele frequencies (1e-5)")
parser.add_argument("-t", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-maf", metavar="FLOAT", type=float, default=0.05,
	help="Threshold for minor allele frequencies")
parser.add_argument("-selection", action="store_true",
	help="Perform PC-based selection scan (Galinsky et al. 2016)")
parser.add_argument("-maf_save", action="store_true",
	help="Save estimated population allele frequencies")
parser.add_argument("-indf_save", action="store_true",
	help="Save estimated individual allele frequencies (Binary)")
parser.add_argument("-bool_save", action="store_true",
	help="Save boolean vector used in MAF filtering (Binary)")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="flash")
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

# Estimate population allele frequencies
def estimateF(D, t=1):
	n, m = D.shape # Dimensions
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
		if nSite == 0:
			f[j] = 0
		else:
			f[j] /= nSite

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

# Center dosages prior to SVD - (E - f)
@jit("void(f4[:, :], f4[:], i8, i8)", nopython=True, nogil=True, cache=True)
def centerE(E, f, S, N):
	n, m = E.shape
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			E[i, j] -= f[j]

# Iteration for estimation of individual allele frequencies
def computeSVD(D, E, f, e, chunks, chunk_N, indf_save):
	# Multithreading - Centering dosages
	threads = [threading.Thread(target=centerE, args=(E, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Reduced SVD of rank K (Scipy library)
	W, s, U = svds(E, k=e)

	# Multithreading - Estimate Pi
	threads = [threading.Thread(target=updateE_SVD, args=(D, E, f, W, s, U, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	if not indf_save:
		U = None

	return W, s, U

# Update E directly from SVD
@jit("void(i1[:, :], f4[:, :], f4[:], f4[:, :], f4[:], f4[:, :], i8, i8)", nopython=True, nogil=True, cache=True)
def updateE_SVD(D, E, f, W, s, U, S, N):
	n, m = E.shape # Dimensions
	K = s.shape[0]
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			if D[i, j] == -9: # Missing site
				E[i, j] = 0.0
				for k in xrange(K):
					E[i, j] += W[i, k]*s[k]*U[k, j]
				E[i, j] += f[j]
				E[i, j] = max(E[i, j], 1e-4)
				E[i, j] = min(E[i, j], 1-(1e-4))
			else:
				E[i, j] = D[i, j]

# Compute individual allele frequencies - add intercept and truncate
@jit("void(f4[:, :], f4[:], i8, i8)", nopython=True, nogil=True, cache=True)
def computePi(Pi, f, S, N):
	n, m = Pi.shape
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			Pi[i, j] += f[j]
			Pi[i, j] = max(Pi[i, j], 1e-4)
			Pi[i, j] = min(Pi[i, j], 1-(1e-4))

# Standardize dosages prior to final SVD - (E - f)/sqrt(f*(1-f))
@jit("void(f4[:, :], f4[:], i8, i8)", nopython=True, nogil=True, cache=True)
def standardizeE(E, f, S, N):
	n, m = E.shape # Dimensions
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			E[i, j] -= f[j]
			E[i, j] /= sqrt(f[j]*(1 - f[j]))

# Final SVD for extracting V and Sigma
def finalSVD(E, f, e, chunks, chunk_N):
	n, m = E.shape # Dimensions

	# Multithreading
	threads = [threading.Thread(target=standardizeE, args=(E, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	V, s, U = svds(E, k=e)
	Sigma = s**2/m
	del U
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

# Selection scan
def galinskyScan(E, V, Sigma, e):
	n, m = E.shape
	Dsquared = np.empty((m, e), dtype=np.float32) # Container for test statistics

	# Loop over different PCs
	for e in xrange(V.shape[1]):
		Dsquared[:, e] = (np.dot(E.T, V[:, e])**2)/Sigma[e]

	return Dsquared


### Main function ###
def flashPCAngsd(D, f, e, indf_save, M=100, M_tole=1e-5, t=1):
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

	if M < 1:
		print "Missingess not taken into account!"

		# Estimate eigenvectors
		print "Inferring set of eigenvector(s)."
		V, Sigma = finalSVD(E, f, e, chunks, chunk_N)
		
		return E, V, Sigma, None
	else:
		# Estimate initial individual allele frequencies
		W, s, U = computeSVD(D, E, f, e, chunks, chunk_N, indf_save)
		prevW = np.copy(W)
		print "Individual allele frequencies estimated (1)"
		
		# Iterative estimation of individual allele frequencies
		for iteration in xrange(2, M+1):
			W, s, U = computeSVD(D, E, f, e, chunks, chunk_N, indf_save)

			# Break iterative update if converged
			diff = rmse(W, prevW, chunks, chunk_N)
			print "Individual allele frequencies estimated (" + str(iteration) + "). RMSE=" + str(diff)
			if diff < M_tole:
				print "Estimation of individual allele frequencies has converged."
				break
			prevW = np.copy(W)

		# Optional construction of individual allele frequencies for saving
		if indf_save:
			Pi = np.empty((n, m), dtype=np.float32)
			Pi = np.dot(W*s, U, out=Pi)
			
			# Multithreading
			threads = [threading.Thread(target=computePi, args=(Pi, f, chunk, chunk_N)) for chunk in chunks]
			for thread in threads:
				thread.start()
			for thread in threads:
				thread.join()
		else:
			Pi = None

		del W, s, U, prevW

		# Estimate eigenvectors
		print "Inferring final set of eigenvector(s)."
		V, Sigma = finalSVD(E, f, e, chunks, chunk_N)
	
		return E, V, Sigma, Pi


### Caller ###
print "FlashPCAngsd Alpha 0.175\n"

# Read in single-read matrix
if args.npy is not None:
	print "Reading in single-read sampling matrix from binary NumPy file."
	# Read from binary NumPy file. Expects np.int8 data format
	D = np.load(args.npy)
	assert D.dtype == np.int8, "NumPy array must be of 8-bit integer format (np.int8)!"
elif args.plink is not None:
	print "Reading PLINK files and converting to single-read sampling matrix."
	# Read from binary PLINK files
	from pysnptools.snpreader import Bed
	import warnings
	warnings.simplefilter(action='ignore', category=FutureWarning)
	readPlink = Bed(args.plink, count_A1=True).read(dtype=np.float32).val
	n, m = readPlink.shape

	# Construct single-read matrix from PLINK files
	D = convertPlink(readPlink, args.t)
	del readPlink
else:
	assert False, "No input file!"

n, m = D.shape

# Population allele frequencies
print "Estimating population allele frequencies."
f = estimateF(D, args.t)

# Removing rare variants
if args.maf > 0.0:
	mask = (f >= args.maf) & (f <= (1 - args.maf))
	print "Filtering variants with a MAF filter of " + str(args.maf) + "."
	f = np.compress(mask, f)
	D = np.compress(mask, D, axis=1)

n, m = D.shape
print str(n) + " samples, " + str(m) + " sites.\n"

# FlashPCAngsd
print "Performing FlashPCAngsd."
print "Using " + str(args.e) + " eigenvector(s)."
E, V, Sigma, Pi = flashPCAngsd(D, f, args.e, args.indf_save, args.m, args.m_tole, args.t)

print "Saving eigenvector(s) as " + args.o + ".eigenvecs.npy (Binary)."
np.save(args.o + ".eigenvecs", V.astype(float, copy=False))
print "Saving eigenvalue(s) as " + args.o + ".eigenvals (Text)."
np.savetxt(args.o + ".eigenvals", Sigma)

if args.selection:
	print "Performing selection scan along each PC."
	Dsquared = galinskyScan(E, V, Sigma, args.e)
	print "Saving test statistics as " + args.o + ".selection.npy (Binary)."
	np.save(args.o + ".selection", Dsquared.astype(float, copy=False))
	del Dsquared, E, V # Clear memory
else:
	del E, V # Clear memory

if args.maf_save:
	print "Saving population allele frequencies as " + args.o + ".maf.npy (Binary)."
	np.save(args.o + ".maf", f.astype(float, copy=False))
	del f # Clear memory

if args.indf_save:
	print "Saving individual allele frequencies as " + args.o + ".pi.npy (Binary)."
	np.save(args.o + ".pi", Pi.astype(float, copy=False))
	del Pi # Clear memory

if (args.bool_save) and (args.maf > 0.0):
	print "Saving boolean vector for used in MAF filtering as " + args.o + ".bool.npy (Binary)"
	np.save(args.o + ".bool", mask)