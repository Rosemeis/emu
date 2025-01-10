import numpy as np
from math import ceil
from time import time
from emu import memory
from emu import shared

##### EMU functions #####
### Read PLINK files
def readPlink(bfile):
	# Find length of fam-file
	N = 0
	with open(f"{bfile}.fam", "r") as fam:
		for _ in fam:
			N += 1
	N_bytes = ceil(N/4) # Length of bytes to describe N individuals

	# Read .bed file
	with open(f"{bfile}.bed", "rb") as bed:
		G = np.fromfile(bed, dtype=np.uint8, offset=3)
	assert (G.shape[0] % N_bytes) == 0, "bim file doesn't match!"
	M = G.shape[0]//N_bytes
	G.shape = (M, N_bytes)
	return G, M, N


### Helper function
# Flip signs of SVD output - Based on scikit-learn (svd_flip)
def signFlip(U, V):
    mcols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[mcols, range(U.shape[1])])
    U *= signs
    V *= signs
    return U, V


### Randomized SVD functions
# SVD through eigendecomposition
def eigSVD(C):
	D, V = np.linalg.eigh(np.dot(C.T, C))
	S = np.sqrt(D)
	U = np.dot(C, V*(1.0/S))
	return np.ascontiguousarray(U[:,::-1]), np.ascontiguousarray(S[::-1]), \
		np.ascontiguousarray(V[:,::-1])

# Randomized SVD with dynamic shift
def randomizedSVD(E, K, power, rng):
	M, N = E.shape
	a = 0.0
	L = K + 10
	A = np.zeros((M, L), dtype=np.float32)
	H = np.zeros((N, L), dtype=np.float32)
	O = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	np.dot(E.T, O, out=H)
	Q, _, _ = eigSVD(H)

	# Power iterations
	for _ in np.arange(power):
		np.dot(E, Q, out=A)
		np.dot(E.T, A, out=H)
		Q, S, _ = eigSVD(H - a*Q)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)
	
	# Extract singular vectors
	np.dot(E, Q, out=A)
	U, S, V = np.linalg.svd(A, full_matrices=False)
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(np.dot(Q, V)[:,:K])
	return U, S[:K], V

# Batched randomized SVD with dynamic shift
def memorySVD(G, U0, V0, f, d, N, K, batch, power, rng):
	M = G.shape[0]
	W = ceil(M/batch)
	a = 0.0
	L = K + 10
	A = np.zeros((M, L), dtype=np.float32)
	H = np.zeros((N, L), dtype=np.float32)
	X = np.zeros((batch, N), dtype=np.float32)
	O = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	for w in np.arange(W):
		M_w = w*batch
		if w == (W-1): # Last batch
			X = np.zeros((M - M_w, N), dtype=np.float32)
		if d is None:
			if U0 is None:
				memory.memCenter(G, X, f, M_w)
			else:
				memory.memCenterSVD(G, U0, V0, X, f, M_w)
		else:
			if U0 is None:
				memory.memFinal(G, X, f, d, M_w)
			else:
				memory.memFinalSVD(G, U0, V0, X, f, d, M_w)
		H += np.dot(X.T, O[M_w:(M_w + X.shape[0])])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for _ in np.arange(power):
		X = np.zeros((batch, N), dtype=np.float32)
		for w in np.arange(W):
			M_w = w*batch
			if w == (W-1): # Last batch
				X = np.zeros((M - M_w, N), dtype=np.float32)
			if d is None:
				if U0 is None:
					memory.memCenter(G, X, f, M_w)
				else:
					memory.memCenterSVD(G, U0, V0, X, f, M_w)
			else:
				if U0 is None:
					memory.memFinal(G, X, f, d, M_w)
				else:
					memory.memFinalSVD(G, U0, V0, X, f, d, M_w)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
		Q, S, _ = eigSVD(H - a*Q)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	X = np.zeros((batch, N), dtype=np.float32)
	for w in np.arange(W):
		M_w = w*batch
		if w == (W-1): # Last batch
			X = np.zeros((M - M_w, N), dtype=np.float32)
		if d is None:
			if U0 is None:
				memory.memCenter(G, X, f, M_w)
			else:
				memory.memCenterSVD(G, U0, V0, X, f, M_w)
		else:
			if U0 is None:
				memory.memFinal(G, X, f, d, M_w)
			else:
				memory.memFinalSVD(G, U0, V0, X, f, d, M_w)
		A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
	U, S, V = np.linalg.svd(A, full_matrices=False)
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(np.dot(Q, V)[:,:K])
	return U, S[:K], V


### EMU algorithm
def emuAlgorithm(G, E, f, d, N, e, K, iter, tole, batch, power, rng):
	M = G.shape[0]

	# Exit without performing EMU
	if iter < 1:
		print("Warning, no EM-PCA iterations are performed!")
		print(f"Extracting {K} eigenvector(s).")
		if E is None:
			U, S, V = memorySVD(G, None, None, f, d, N, K, batch, power, rng)
		else:
			shared.standardInit(G, E, f, d)
			U, S, V = randomizedSVD(E, K, power, rng)
		U, V = signFlip(U, V)
		return U, S, V, 0, False
	else:
		# Set up acceleration containers
		U1 = np.zeros((M, e), dtype=np.float32)
		U2 = np.zeros((M, e), dtype=np.float32)
		V1 = np.zeros((N, e), dtype=np.float32)
		V2 = np.zeros((N, e), dtype=np.float32)

		# Estimate initial individual allele frequencies
		print("Initiating accelerated EM scheme")
		if E is None:
			U, S, V = memorySVD(G, None, None, f, None, N, e, batch, power, rng)
		else:
			shared.centerInit(G, E, f)
			U, S, V = randomizedSVD(E, K, power, rng)
		U, V = signFlip(U, V)
		V *= S
		U_pre = np.copy(U)

		# Iterative estimation of individual allele frequencies
		ts = time()
		for it in range(1, iter+1):
			# 1st SVD step
			if E is None:
				U1, S1, V1 = memorySVD(G, U, V, f, None, N, e, batch, power, rng)
			else:
				shared.centerAccel(G, E, U, V, f)
				U1, S1, V1 = randomizedSVD(E, e, power, rng)
			U1, V1 = signFlip(U1, V1)
			V1 *= S1

			# 2nd SVD step
			if E is None:
				U2, S2, V2 = memorySVD(G, U1, V1, f, None, N, e, batch, power, rng)
			else:
				shared.centerAccel(G, E, U1, V1, f)
				U2, S2, V2 = randomizedSVD(E, e, power, rng)
			U2, V2 = signFlip(U2, V2)
			V2 *= S2

			# QN steps
			shared.alphaStep(U, U1, U2)
			shared.alphaStep(V, V1, V2)

			# Stabilization step
			if E is None:
				U, S, V = memorySVD(G, U, V, f, None, N, e, batch, power, rng)
			else:
				shared.centerAccel(G, E, U, V, f)
				U, S, V = randomizedSVD(E, e, power, rng)
			U, V = signFlip(U, V)
			V *= S

			# Break iterative update if converged
			rmseU = shared.rmse(U, U_pre)
			print(f"({it})\tRMSE = {rmseU:.8f}\t({time()-ts:.1f}s)")
			if rmseU < tole:
				print("EM-PCA has converged.")
				converged = True
				break
			if it == iter:
				print("EM-PCA did not converge!")
				converged = False
			memoryview(U_pre.ravel())[:] = memoryview(U.ravel())
			ts = time()
		del U1, U2, U_pre, V1, V2, S1, S2

		# Estimating final SVD
		print(f"Extracting {K} eigenvector(s).")
		if E is None:
			U, S, V = memorySVD(G, U, V, f, d, N, K, batch, power, rng)
		else:
			shared.standardAccel(G, E, U, V, f, d)
			U, S, V = randomizedSVD(E, K, power, rng)
		U, V = signFlip(U, V)
		return U, S, V, it, converged
	