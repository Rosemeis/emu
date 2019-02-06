"""
Our FlashPCAngsd approach.
Jonas Meisner, Siyang Liu and Anders Albrechtsen (2018)
"""

__author__ = "Jonas Meisner"

# Libraries and modules
import numpy as np
import threading
import argparse
from tqdm import tqdm
from numba import jit
from math import sqrt
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

##### Argparse #####
parser = argparse.ArgumentParser(prog="FlashPCAngsd")
parser.add_argument("--version", action="version", version="%(prog)s alpha 0.2")
parser.add_argument("-npy", metavar="FILE",
	help="Input file (.npy)")
parser.add_argument("-plink", metavar="PREFIX",
	help="Prefix for binary PLINK files")
parser.add_argument("-e", metavar="INT", type=int,
	help="Number of eigenvectors to use in IAF estimation")
parser.add_argument("-k", metavar="INT", type=int,
	help="Number of eigenvectors to output in final SVD")
parser.add_argument("-m", metavar="INT", type=int, default=100,
	help="Maximum iterations for estimation of individual allele frequencies (100)")
parser.add_argument("-m_tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for update in estimation of individual allele frequencies (1e-5)")
parser.add_argument("-t", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-maf", metavar="FLOAT", type=float, default=0.05,
	help="Threshold for minor allele frequencies")
parser.add_argument("-chunks", metavar="INT", type=int, default=10,
	help="Number of chunks used for reading PLINK bed file (10)")
parser.add_argument("-selection", action="store_true",
	help="Perform PC-based selection scan (Galinsky et al. 2016)")
parser.add_argument("-accel", action="store_true",
	help="Perform accelerated EM-PCA")
parser.add_argument("-pop", metavar="FILE",
	help="Input file of region-based allele frequencies (.npy)")
parser.add_argument("-reg", metavar="FILE",
	help="Region index vector for individuals (0-indexed)")
parser.add_argument("-svd", metavar="STRING", default="arpack",
	help="Method for performing truncated SVD (ARPACK/Randomized)")
parser.add_argument("-randomized_power", metavar="INT", type=int, default=4,
	help="Number of power iterations in randomized SVD")
parser.add_argument("-maf_save", action="store_true",
	help="Save estimated population allele frequencies")
parser.add_argument("-bool_save", action="store_true",
	help="Save boolean vector used in MAF filtering (Binary)")
parser.add_argument("-indf_save", action="store_true",
	help="Save estimated individual allele frequencies")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="flash")
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

# Update E - initial step (Region-based initialization)
@jit("void(i1[:, :], f4[:, :], i1[:], i8, i8, f4[:, :])", nopython=True, nogil=True, cache=True)
def updateE_multi_init(D, F, Pvec, S, N, E):
	n, m = E.shape # Dimensions
	K = F.shape[1]
	for i in xrange(S, min(S+N, n)):
		# Estimate posterior probabilities and update dosages
		for j in xrange(m):
			for k in xrange(K):
				if Pvec[i] == k: 
					if D[i, j] == -9: # Missing site
						E[i, j] = F[j, k]
					else:
						E[i, j] = D[i, j]

# Center dosages prior to SVD - (E - f)
@jit("void(f4[:, :], f4[:], i8, i8)", nopython=True, nogil=True, cache=True)
def centerE(E, f, S, N):
	n, m = E.shape # Dimensions
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			E[i, j] -= f[j]

# Iteration for estimation of individual allele frequencies
def computeSVD(D, E, f, e, chunks, chunk_N, accel, method, power):
	# Multithreading - Centering dosages
	threads = [threading.Thread(target=centerE, args=(E, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	if method == "arpack": # Reduced SVD of rank K (Scipy library)
		W, s, U = svds(E, k=e)
	elif method == "randomized": # Scikit-learn implementation
		W, s, U = randomized_svd(E, e, n_iter=power)

	# Multithreading - Estimate Pi
	threads = [threading.Thread(target=updateE_SVD, args=(D, E, f, W, s, U, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	if not accel:
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
	n, m = Pi.shape # Dimensions
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
def finalSVD(E, f, e, chunks, chunk_N, method, power):
	n, m = E.shape # Dimensions

	# Multithreading
	threads = [threading.Thread(target=standardizeE, args=(E, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	if method == "arpack":
		V, s, U = svds(E, k=e)
		V, s, U = V[:, ::-1], s[::-1], U[::-1, :]
	elif method == "randomized":
		V, s, U = randomized_svd(E, e, n_iter=power)

	return V, s, U

# Measure difference
def rmse(A, B, chunks, chunk_N):
	n, m = A.shape # Dimensions
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
	n, m = A.shape # Dimensions
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			if np.sign(A[i, j]) == np.sign(B[i,j]):
				R[i] += (A[i, j] - B[i, j])*(A[i, j] - B[i, j])
			else:
				C = A[i, j]*-1
				R[i] += (C - B[i, j])*(C - B[i, j])			

# Selection scan
def galinskyScan(U, e):
	_, m = U.shape # Dimensions
	Dsquared = np.empty((m, e), dtype=np.float32) # Container for test statistics

	# Loop over different PCs
	for k in xrange(e):
		Dsquared[:, e] = (U[e]**2)*m

	return Dsquared

# Acceleration
def squarem_alpha(E0, W1, s1, U1, W2, s2, U2, chunks, chunk_N):
	n, _ = W1.shape
	sr2 = np.zeros(n, dtype=np.float32)
	sv2 = np.zeros(n, dtype=np.float32)

	# Multithreading
	threads = [threading.Thread(target=squarem_alpha_inner, args=(E0, W1, s1, U1, W2, s2, U2, chunk, chunk_N, sr2, sv2)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()
	
	sr2sum = np.sum(sr2)
	sv2sum = np.sum(sv2)
	alpha = sqrt(sr2sum/sv2sum)

	return alpha, sr2sum, sv2sum

@jit("void(f4[:, :], f4[:, :], f4[:], f4[:, :], f4[:, :], f4[:], f4[:, :], i8, i8, f4[:], f4[:])", nopython=True, nogil=True, cache=True)
def squarem_alpha_inner(E0, W1, s1, U1, W2, s2, U2, S, N, sr2, sv2):
	n, K = W1.shape
	m, _ = U1.shape
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			theta1 = 0
			theta2 = 0
			for k in xrange(K):
				theta1 += W1[i, k]*s1[k]*U1[k, j]
				theta2 += W2[i, k]*s2[k]*U2[k, j]
			sr2[i] += (theta1 - E0[i, j])**2
			sv2[i] += (theta2 - 2*theta1 + E0[i, j])**2

@jit("void(i1[:, :], f4[:, :], f4[:], f4[:, :], f4[:, :], f4[:], f4[:, :], f4[:, :], f4[:], f4[:, :], f8, i8, i8)", nopython=True, nogil=True, cache=True)
def accelUpdate(D, E, f, E0, W1, s1, U1, W2, s2, U2, alpha, S, N):
	n, K = W1.shape
	m, _ = U1.shape
	for i in xrange(S, min(S+N, n)):
		for j in xrange(m):
			theta1 = 0
			theta2 = 0
			for k in xrange(K):
				theta1 += W1[i, k]*s1[k]*U1[k, j]
				theta2 += W2[i, k]*s2[k]*U2[k, j]
			E0[i, j] = E0[i, j] + 2*alpha*(theta1 - E0[i, j]) + alpha*alpha*(theta2 - 2*theta1 + E0[i, j])
			if D[i, j] == -9:
				E[i, j] = E0[i, j] + f[j]
				E[i, j] = max(E[i, j], 1e-4)
				E[i, j] = min(E[i, j], 1-(1e-4))
			else:
				E[i, j] = D[i, j]


### Main function ###
def flashPCAngsd(D, f, e, K, accel, F=None, Pvec=None, M=100, M_tole=1e-5, method="arpack", svd_iter=2, t=1):
	n, m = D.shape # Dimensions
	E = np.empty((n, m), dtype=np.float32) # Initiate E

	# Multithreading parameters
	chunk_N = int(np.ceil(float(n)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	if F is None:
		# Multithreading
		threads = [threading.Thread(target=updateE_init, args=(D, f, chunk, chunk_N, E)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
	else:
		# Multithreading
		threads = [threading.Thread(target=updateE_multi_init, args=(D, F, Pvec, chunk, chunk_N, E)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		del F, Pvec

	if M < 1:
		print "Missingess not taken into account!"

		# Estimate eigenvectors
		print "Inferring set of eigenvector(s)."
		V, Sigma, U = finalSVD(E, f, K, chunks, chunk_N, method, svd_iter)
		
		return E, V, Sigma, U
	else:
		if not accel:
			# Estimate initial individual allele frequencies
			W, s, U = computeSVD(D, E, f, e, chunks, chunk_N, accel, method, svd_iter)
			prevW = np.copy(W)
			print "Individual allele frequencies estimated (1)"
			
			# Iterative estimation of individual allele frequencies
			for iteration in xrange(2, M+1):
				W, s, U = computeSVD(D, E, f, e, chunks, chunk_N, accel, method, svd_iter)

				# Break iterative update if converged
				diff = rmse(W, prevW, chunks, chunk_N)
				print "Individual allele frequencies estimated (" + str(iteration) + "). RMSE=" + str(diff)
				if diff < M_tole:
					print "Estimation of individual allele frequencies has converged."
					break
				prevW = np.copy(W)
		
		else: # Accelerated EM (SQUAREM2)
			minStep, maxStep = 1.0, 1.0
			W0, s0, U0 = computeSVD(D, E, f, e, chunks, chunk_N, accel, method, svd_iter)
			E0 = np.dot(W0*s0, U0)
			del W0, s0, U0

			for iteration in xrange(1, M+1):
				W1, s1, U1 = computeSVD(D, E, f, e, chunks, chunk_N, accel, method, svd_iter)
				W2, s2, U2 = computeSVD(D, E, f, e, chunks, chunk_N, accel, method, svd_iter)
				alpha, sr2, sv2 = squarem_alpha(E0, W1, s1, U1, W2, s2, U2, chunks, chunk_N)
				alpha = max(minStep, min(maxStep, alpha))
				if alpha == maxStep:
					maxStep = 4*maxStep

				# Multithreading
				threads = [threading.Thread(target=accelUpdate, args=(D, E, f, E0, W1, s1, U1, W2, s2, U2, alpha, chunk, chunk_N)) for chunk in chunks]
				for thread in threads:
					thread.start()
				for thread in threads:
					thread.join()

				print "Iteration " + str(iteration) + ": " + str([alpha, sr2, sv2])
				if sr2 <= M_tole:
					break

		if not accel:
			del W, s, U, prevW
		else:
			del E0, W1, s1, U1, W2, s2, U2

		# Estimate eigenvectors
		print "Inferring final set of eigenvector(s)."
		V, s, U = finalSVD(E, f, K, chunks, chunk_N, method, svd_iter)
		del E
	
		return V, s, U


### Caller ###
print "FlashPCAngsd Alpha 0.2\n"

# Workflow check
svdlist = ["arpack", "randomized"]
assert args.svd in svdlist, "Must use a valid truncated SVD approach!"

if args.k is None:
	K = args.e
else:
	K = args.k

# Read in single-read matrix
if args.npy is not None:
	print "Reading in single-read sampling matrix from binary NumPy file."
	# Read from binary NumPy file. Expects np.int8 data format
	D = np.load(args.npy)
	assert D.dtype == np.int8, "NumPy array must be of 8-bit integer format (np.int8)!"
elif args.plink is not None:
	print "Reading PLINK files and converting to single-read sampling matrix."
	# Read from binary PLINK files
	from pandas_plink import read_plink
	_, _, G = read_plink(args.plink)
	m, n = G.shape

	# Construct single-read matrix from PLINK files
	D = np.empty((n, m), dtype=np.int8)

	# Loop over chunks of bed file
	idx_split = np.array_split(np.arange(m, dtype=int), args.chunks)
	for c in tqdm(range(args.chunks), desc="Reading chunks"):
		G_chunk = G[idx_split[c], :].compute().T
		D[:, idx_split[c]] = convertPlink(G_chunk, args.t)
	del G, G_chunk
else:
	assert False, "No input file!"

n, m = D.shape

# Population allele frequencies
print "Estimating population allele frequencies."
f = estimateF(D, args.t)

if args.pop is not None:
	print "Loading region-specific allele frequencies."
	assert args.reg is not None, "Regional vector of individuals must also be supplied!"
	F = np.load(args.pop)
	Pvec = np.genfromtxt(args.reg, dtype=np.int8)
else:
	F = None
	Pvec = None

# Removing rare variants
if args.maf > 0.0:
	mask = (f >= args.maf) & (f <= (1 - args.maf))
	print "Filtering variants with a MAF filter of " + str(args.maf) + "."
	f = np.compress(mask, f)
	D = np.compress(mask, D, axis=1)
	if args.pop is not None:
		F = np.compress(mask, F, axis=0)
		assert F.shape[0] == f.shape[0], "Dimensions of frequencies doesn't fit!"

n, m = D.shape
print str(n) + " samples, " + str(m) + " sites.\n"

# FlashPCAngsd
print "Performing FlashPCAngsd."
print "Using " + str(args.e) + " eigenvector(s)."
V, s, U = flashPCAngsd(D, f, args.e, K, args.accel, F, Pvec, args.m, args.m_tole, args.svd, args.randomized_power, args.t)

print "Saving eigenvector(s) as " + args.o + ".eigenvecs.npy (Binary)."
np.save(args.o + ".eigenvecs", V.astype(float, copy=False))
print "Saving eigenvalue(s) as " + args.o + ".eigenvals (Text)."
np.savetxt(args.o + ".eigenvals", s**2/m)

if args.indf_save:
	print "Saving individual allele frequencies as " + args.o + ".indf.npy (Binary)."
	np.save(args.o + ".indf", np.dot(V*s, U).astype(float, copy=False))
del V, s # Clear memory

if args.selection:
	print "Performing selection scan along each PC."
	Dsquared = galinskyScan(U, args.e)
	print "Saving test statistics as " + args.o + ".selection.npy (Binary)."
	np.save(args.o + ".selection", Dsquared.astype(float, copy=False))
	del Dsquared # Clear memory
del U # Clear memory

if args.maf_save:
	print "Saving population allele frequencies as " + args.o + ".maf.npy (Binary)."
	np.save(args.o + ".maf", f.astype(float, copy=False))
del f # Clear memory

if (args.bool_save) and (args.maf > 0.0):
	print "Saving boolean vector for used in MAF filtering as " + args.o + ".bool.npy (Binary)"
	np.save(args.o + ".bool", mask.astype(int, copy=False))