"""
Cython implementation of EMU-mem (Memory based). Data matrix is stored in special 2-bit format.
Performs iterative SVD of allele count matrix (EM-PCA) based on custom Halko method.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen

Example usage: python emu_mem.py -plink fileprefix -e 2 -t 64 -accel -o flash
"""

__author__ = "Jonas Meisner"

import halko
import shared
import numpy as np
import argparse
import subprocess
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from math import ceil

### Find length of PLINK files
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, err = process.communicate()
	return int(result.split()[0])

### Range finder functions of Q
def range_finder(D, f, e, F, p, Bi, n, m, svd_power, t):
	K = e + 10
	C = np.zeros((m, K), dtype=np.float32)

	# Sample Gaussian vectors
	np.random.seed(0)
	Q = np.random.normal(size=(n, K)).astype(np.float32, copy=False)

	# Power iterations
	for pow_i in range(svd_power):
		if F is None:
			C.fill(0)
			halko.matMul_Freq(D, f, Q, C, Bi, n, m, t)
			Q.fill(0)
			halko.matMulTrans_Freq(D, f, C, Q, Bi, n, m, t)
		else:
			C.fill(0)
			halko.matMul_Guide(D, f, F, p, Q, C, Bi, n, m, t)
			Q.fill(0)
			halko.matMulTrans_Guide(D, f, F, p, C, Q, Bi, n, m, t)
	C.fill(0)
	if F is None:
		halko.matMul_Freq(D, f, Q, C, Bi, n, m, t)
	else:
		halko.matMul_Guide(D, f, F, p, Q, C, Bi, n, m, t)
	Q, _ = linalg.qr(C, mode='economic')
	return Q

# Range finder of Q when mapping back to domain for E=WSU.T
def range_finder_domain(D, f, e, U, s, W, Bi, n, m, svd_power, t):
	K = e + 10
	C = np.zeros((m, K), dtype=np.float32)

	# Sample Gaussian vectors
	np.random.seed(0)
	Q = np.random.normal(size=(n, K)).astype(np.float32, copy=False)

	# Power iterations
	for pow_i in range(svd_power):
		C.fill(0)
		halko.matMul_SVD_domain(D, f, U, s, W, Q, C, Bi, n, m, t)
		Q.fill(0)
		halko.matMulTrans_SVD_domain(D, f, U, s, W, C, Q, Bi, n, m, t)
	C.fill(0)
	halko.matMul_SVD_domain(D, f, U, s, W, Q, C, Bi, n, m, t)
	Q, _ = linalg.qr(C, mode='economic')
	return Q

# Range finder of Q for final iteration
def range_finder_final(D, f, e, F, p, U, s, W, Bi, n, m, svd_power, t):
	K = e + 10
	C = np.zeros((m, K), dtype=np.float32)

	# Sample Gaussian vectors
	np.random.seed(0)
	Q = np.random.normal(size=(n, K)).astype(np.float32, copy=False)

	# Power iterations
	for pow_i in range(svd_power):
		if (F is None) & (W is None):
			C.fill(0)
			halko.matMulFinal_Freq(D, f, Q, C, Bi, n, m, t)
			C, _ = linalg.lu(C, permute_l=True)
			Q = np.zeros((n, K), dtype=np.float32)
			halko.matMulTransFinal_Freq(D, f, C, Q, Bi, n, m, t)
			Q, _ = linalg.lu(Q, permute_l=True)
		elif (W is None):
			C.fill(0)
			halko.matMulFinal_Guide(D, f, F, p, Q, C, Bi, n, m, t)
			C, _ = linalg.lu(C, permute_l=True)
			Q = np.zeros((n, K), dtype=np.float32)
			halko.matMulTransFinal_Guide(D, f, F, p, C, Q, Bi, n, m, t)
			Q, _ = linalg.lu(Q, permute_l=True)
		else:
			C.fill(0)
			halko.matMulFinal_SVD(D, f, U, s, W, Q, C, Bi, n, m, t)
			C, _ = linalg.lu(C, permute_l=True)
			Q = np.zeros((n, K), dtype=np.float32)
			halko.matMulTransFinal_SVD(D, f, U, s, W, C, Q, Bi, n, m, t)
			Q, _ = linalg.lu(Q, permute_l=True)
	C.fill(0)
	if (F is None) & (W is None):
		halko.matMulFinal_Freq(D, f, Q, C, Bi, n, m, t)
	elif (W is None):
		halko.matMulFinal_Guide(D, f, F, p, Q, C, Bi, n, m, t)
	else:
		halko.matMulFinal_SVD(D, f, U, s, W, Q, C, Bi, n, m, t)
	Q, _ = linalg.qr(C, mode='economic')
	return Q

# Acceleration - Range finder of Q when mapping back to domain for E=USW.T
def range_finder_domain_accel(D, f, e, U, W, Bi, n, m, svd_power, t):
	K = e + 10
	C = np.zeros((m, K), dtype=np.float32)

	# Sample Gaussian vectors
	np.random.seed(0)
	Q = np.random.normal(size=(n, K)).astype(np.float32, copy=False)

	# Power iterations
	for pow_i in range(svd_power):
		C.fill(0)
		halko.matMul_SVD_domain_accel(D, f, U, W, Q, C, Bi, n, m, t)
		Q.fill(0)
		halko.matMulTrans_SVD_domain_accel(D, f, U, W, C, Q, Bi, n, m, t)
	C.fill(0)
	halko.matMul_SVD_domain_accel(D, f, U, W, Q, C, Bi, n, m, t)
	Q, _ = linalg.qr(C, mode='economic')
	return Q

### Iterative SVD functions
def customSVD(D, f, e, F, p, Bi, n, m, svd_power, t):
	Q = range_finder(D, f, e, F, p, Bi, n, m, svd_power, t)
	Bt = np.zeros((n, Q.shape[1]), dtype=np.float32)

	# B.T = dot(E.T, Q)
	if F is None:
		halko.matMulTrans_Freq(D, f, Q, Bt, Bi, n, m, t)
	else:
		halko.matMulTrans_Guide(D, f, F, p, Q, Bt, Bi, n, m, t)

	# SVD on thin matrix
	Uhat, s, V = linalg.svd(Bt.T, full_matrices=False)
	del Bt
	U = np.dot(Q, Uhat)

	# Correct sign
	U, V = svd_flip(U, V)
	return U[:,:e], s[:e], V[:e,:]

# Map to domain SVD
def customDomainSVD(D, f, e, U, s, W, Bi, n, m, svd_power, t):
	Q = range_finder_domain(D, f, e, U, s, W, Bi, n, m, svd_power, t)
	Bt = np.zeros((n, Q.shape[1]), dtype=np.float32)

	# B.T = dot(E.T, Q)
	halko.matMulTrans_SVD_domain(D, f, U, s, W, Q, Bt, Bi, n, m, t)

	# SVD on thin matrix
	Uhat, s, V = linalg.svd(Bt.T, full_matrices=False)
	del Bt
	U = np.dot(Q, Uhat)

	# Correct sign
	U, V = svd_flip(U, V)
	return U[:,:e], s[:e], V[:e,:]

# Final SVD
def customFinalSVD(D, f, e, F, p, U, s, W, Bi, n, m, svd_power, t):
	Q = range_finder_final(D, f, e, F, p, U, s, W, Bi, n, m, svd_power, t)
	Bt = np.zeros((n, Q.shape[1]), dtype=np.float32)

	# B.T = dot(E.T, Q)
	if (F is None) & (W is None):
		halko.matMulTransFinal_Freq(D, f, Q, Bt, Bi, n, m, t)
	elif (W is None):
		halko.matMulTransFinal_Guide(D, f, F, p, Q, Bt, Bi, n, m, t)
	else:
		halko.matMulTransFinal_SVD(D, f, U, s, W, Q, Bt, Bi, n, m, t)

	# SVD on thin matrix
	Uhat, s, V = linalg.svd(Bt.T, full_matrices=False)
	del Bt
	U = np.dot(Q, Uhat)

	# Correct sign
	U, V = svd_flip(U, V)
	return U[:,:e], s[:e], V[:e,:]

# Acceleration - Map to domain SVD
def customDomainSVD_accel(D, f, e, U, W, Bi, n, m, svd_power, t):
	Q = range_finder_domain_accel(D, f, e, U, W, Bi, n, m, svd_power, t)
	Bt = np.zeros((n, Q.shape[1]), dtype=np.float32)

	# B.T = dot(E.T, Q)
	halko.matMulTrans_SVD_domain_accel(D, f, U, W, Q, Bt, Bi, n, m, t)

	# SVD on thin matrix
	Uhat, s, V = linalg.svd(Bt.T, full_matrices=False)
	del Bt
	U = np.dot(Q, Uhat)

	# Correct sign
	U, V = svd_flip(U, V)
	return U[:,:e], s[:e], V[:e,:]


### Main function ###
def emuMemory(D, f, e, K, M, M_tole, F, p, U, s, W, Bi, n, m, svd_power, indf_save, output, accel, t):
	if accel:
		print("Using accelerated EM scheme (SqS3)")
		diffU_1 = np.empty((m, e), dtype=np.float32)
		diffU_2 = np.empty((m, e), dtype=np.float32)
		diffU_3 = np.empty((m, e), dtype=np.float32)
		diffW_1 = np.empty((e, n), dtype=np.float32)
		diffW_2 = np.empty((e, n), dtype=np.float32)
		diffW_3 = np.empty((e, n), dtype=np.float32)
		maxU, maxW = 1.0, 1.0

	if M < 1:
		print("Warning, no EM-PCA iterations are performed!")
		print("Inferring set of eigenvector(s).")
		U, s, V = customFinalSVD(D, f, e, F, p, U, s, W, Bi, n, m, svd_power, t)
		return U, s, V
	else:
		# Estimate initial individual allele frequencies
		if accel:
			print("Initiating accelerated EM scheme (1)")
		if (W is None):
			U, s, W = customSVD(D, f, e, F, p, Bi, n, m, svd_power, t)
		else:
			U, s, W = customDomainSVD(D, f, e, U, s, W, Bi, n, m, svd_power, t)
		if not accel:
			print("Individual allele frequencies estimated (1).")
		else:
			W = W*s.reshape((e, 1))
		prevU = np.copy(U)

		# Iterative estimation of individual allele frequencies
		for iteration in range(2, M+1):
			if accel:
				U1, s1, W1 = customDomainSVD_accel(D, f, e, U, W, Bi, n, m, svd_power, t)
				W1 = W1*s1.reshape((e, 1))
				shared.matMinus(U1, U, diffU_1)
				shared.matMinus(W1, W, diffW_1)
				sr2_U = shared.matSumSquare(diffU_1)
				sr2_W = shared.matSumSquare(diffW_1)
				U2, s2, W2 = customDomainSVD_accel(D, f, e, U1, W1, Bi, n, m, svd_power, t)
				W2 = W2*s2.reshape((e, 1))
				shared.matMinus(U2, U1, diffU_2)
				shared.matMinus(W2, W1, diffW_2)

				# SQUAREM update of W and U SqS3
				shared.matMinus(diffU_2, diffU_1, diffU_3)
				shared.matMinus(diffW_2, diffW_1, diffW_3)
				sv2_U = shared.matSumSquare(diffU_3)
				sv2_W = shared.matSumSquare(diffW_3)
				alpha_U = max(1.0, np.sqrt(sr2_U/sv2_U))
				alpha_W = max(1.0, np.sqrt(sr2_W/sv2_W))

				# New accelerated update
				shared.matUpdate(U, diffU_1, diffU_3, alpha_U)
				shared.matUpdate(W, diffW_1, diffW_3, alpha_W)
			else:
				U, s, W = customDomainSVD(D, f, e, U, s, W, Bi, n, m, svd_power, t)

			# Break iterative update if converged
			diff = np.sqrt(np.sum(shared.rmse(U, prevU, t))/(m*e))
			print("Individual allele frequencies estimated (" + str(iteration) + "). RMSE=" + str(diff))
			if diff < M_tole:
				print("Estimation of individual allele frequencies has converged.")
				break
			prevU = np.copy(U)
		del prevU

		if accel:
			U, s, W = customDomainSVD_accel(D, f, e, U, W, Bi, n, m, svd_power, t)
			del U1, U2, s1, s2, W1, W2, diffU_1, diffU_2, diffU_3, diffW_1, diffW_2, diffW_3

		if indf_save:
			print("Saving singular matrices for future use (.w.npy, .s.npy, .u.npy).")
			np.save(output + ".w", W)
			np.save(output + ".s", s)
			np.save(output + ".u", U)

		# Estimating SVD
		print("Inferring set of eigenvector(s).")
		U, s, V = customFinalSVD(D, f, K, F, p, U, s, W, Bi, n, m, svd_power, t)
		del W
		
		return U, s, V


##### Argparse #####
parser = argparse.ArgumentParser(prog="EMU-mem")
parser.add_argument("--version", action="version", version="%(prog)s alpha 0.65")
parser.add_argument("-plink", metavar="FILE-PREFIX",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-e", metavar="INT", type=int,
	help="Number of eigenvectors to use in IAF estimation")
parser.add_argument("-k", metavar="INT", type=int,
	help="Number of eigenvectors to output in final SVD")
parser.add_argument("-m", metavar="INT", type=int, default=100,
	help="Maximum iterations for estimation of individual allele frequencies (100)")
parser.add_argument("-m_tole", metavar="FLOAT", type=float, default=5e-7,
	help="Tolerance for update in estimation of individual allele frequencies (5e-7)")
parser.add_argument("-t", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-selection", action="store_true",
	help="Perform PC-based selection scan (Galinsky et al. 2016)")
parser.add_argument("-maf_save", action="store_true",
	help="Save estimated population allele frequencies")
parser.add_argument("-indf_save", action="store_true",
	help="Save estimated singular matrices")
parser.add_argument("-index", metavar="FILE",
	help="Index for guided allele frequencies")
parser.add_argument("-svd_power", metavar="INT", type=int, default=3,
	help="Number of power iterations in randomized SVD")
parser.add_argument("-w", metavar="FILE",
	help="Left singular matrix (.w.npy)")
parser.add_argument("-s", metavar="FILE",
	help="Singular values (.s.npy)")
parser.add_argument("-u", metavar="FILE",
	help="Right singular matrix (.u.npy)")
parser.add_argument("-accel", action="store_true",
	help="Accelerated EM")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output name", default="emu_mem")
args = parser.parse_args()


### Caller ###
print("EMU-mem 0.65\n")

# Set K
if args.k is None:
	K = args.e
else:
	K = args.k

# Read in data to memory (.bed file -> uint8)
print("Reading in single-read sampling matrix from PLINK files.")
assert args.plink is not None, "No valid input given! Must use '-plink'!"
# Finding length of .fam and .bim file and read .bed file into NumPy array
n = extract_length(args.plink + ".fam")
m = extract_length(args.plink + ".bim")
bed = open(args.plink + ".bed", "rb")
B = np.fromfile(bed, dtype=np.uint8, offset=3)
bed.close()

Bi = ceil(n/4) # Length of bytes to describe n individuals
D = B.reshape((m, Bi))
print(str(n) + " samples, " + str(m) + " sites.\n")

# Estimate population allele frequencies
print("Estimating population allele frequencies.")
f = np.zeros(m, dtype=np.float32)
halko.estimateF(D, f, Bi, n, m, args.t)

# Guided meta allele frequencies
if (args.index is not None) & (args.w is None):
	print("Estimating guided allele frequencies.")
	p = np.load(args.index)
	F = np.zeros((m, max(p)+1), dtype=np.float32)
	halko.estimateF_guided(D, f, F, p, Bi, n, m, args.t)
else:
	p, F = None, None

# Use eigenvectors from previous run
if args.w is not None:
	assert args.s is not None, "Must supply both -s and -u along with -w!"
	assert args.u is not None, "Must supply both -s and -u along with -w!"
	print("Reading singular matrices from previous run.")
	W = np.load(args.w)
	assert W.shape[0] == n, "Number of samples in W must match data!"
	s = np.load(args.s)
	U = np.load(args.u)
	assert U.shape[1] == m, "Number of sites in U must match data!"
else:
	W, s, U = None, None, None

# Ultramem
print("Performing EMU-mem variant.")
print("Using " + str(args.e) + " eigenvector(s).")
U, s, V = emuMemory(D, f, args.e, K, args.m, args.m_tole, F, p, U, s, W, Bi, n, m, args.svd_power, \
	args.indf_save, args.o, args.accel, args.t)

print("Saving eigenvector(s) as " + args.o + ".eigenvecs.npy (Binary).")
np.save(args.o + ".eigenvecs", V.T.astype(float, copy=False))
print("Saving eigenvalue(s) as " + args.o + ".eigenvals (Text).")
np.savetxt(args.o + ".eigenvals", s**2/m)

if args.selection:
	print("Performing selection scan along each PC.")
	Dsquared = np.zeros((m, args.e), dtype=np.float32)
	halko.galinskyScan(U[:,:args.e], Dsquared, m, args.e, args.t)
	print("Saving test statistics as " + args.o + ".selection.npy (Binary).")
	np.save(args.o + ".selection", Dsquared.astype(float, copy=False))
	del Dsquared # Clear memory
del U # Clear memory

if args.maf_save:
	print("Saving population allele frequencies as " + args.o + ".maf.npy (Binary).")
	np.save(args.o + ".maf", f.astype(float, copy=False))
del f # Clear memory