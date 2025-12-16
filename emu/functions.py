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
def eigSVD(H):
	D, V = np.linalg.eigh(np.dot(H.T, H))
	S = np.sqrt(D)
	U = np.dot(H, V*(1.0/S))
	return np.ascontiguousarray(U[:,::-1]), np.ascontiguousarray(S[::-1]), np.ascontiguousarray(V[:,::-1])

# Randomized SVD with dynamic shift
def randomizedSVD(E, K, power, rng):
	M, N = E.shape
	a = 0.0
	L = max(K + 10, 20)
	H = np.zeros((N, L), dtype=np.float32)
	A = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	np.dot(E.T, A, out=H)
	Q, _, _ = eigSVD(H)

	# Power iterations
	for _ in np.arange(power):
		np.dot(E, Q, out=A)
		np.dot(E.T, A, out=H)
		H -= a*Q
		Q, S, _ = eigSVD(H)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)
	
	# Extract singular vectors
	np.dot(E, Q, out=A)
	U, S, V = eigSVD(A)
	return np.ascontiguousarray(U[:,:K]), S[:K], np.ascontiguousarray(np.dot(Q, V)[:,:K])

# Batched randomized SVD with dynamic shift (frequency-based)
def memoryInit(G, f, d, N, K, batch, power, rng):
	M = G.shape[0]
	W = ceil(M/batch)
	a = 0.0
	L = max(K + 10, 20)
	H = np.zeros((N, L), dtype=np.float32)
	X = np.zeros((batch, N), dtype=np.float32)
	A = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	for w in np.arange(W):
		M_w = w*batch
		if w == (W - 1): # Last batch
			X = np.zeros((M - M_w, N), dtype=np.float32)
		memory.memCen(G, X, f, M_w) if d is None else memory.memFin(G, X, f, d, M_w)
		H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for _ in np.arange(power):
		X = np.zeros((batch, N), dtype=np.float32)
		for w in np.arange(W):
			M_w = w*batch
			if w == (W - 1): # Last batch
				X = np.zeros((M - M_w, N), dtype=np.float32)
			memory.memCen(G, X, f, M_w) if d is None else memory.memFin(G, X, f, d, M_w)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
		H -= a*Q
		Q, S, _ = eigSVD(H)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	X = np.zeros((batch, N), dtype=np.float32)
	for w in np.arange(W):
		M_w = w*batch
		if w == (W - 1): # Last batch
			X = np.zeros((M - M_w, N), dtype=np.float32)
		memory.memCen(G, X, f, M_w) if d is None else memory.memFin(G, X, f, d, M_w)
		A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
	U, S, V = eigSVD(A)
	return np.ascontiguousarray(U[:,:K]), S[:K], np.ascontiguousarray(np.dot(Q, V)[:,:K])

# Batched randomized SVD with dynamic shift
def memorySVD(G, U0, V0, f, d, N, K, batch, power, rng):
	M = G.shape[0]
	W = ceil(M/batch)
	a = 0.0
	L = max(K + 10, 20)
	H = np.zeros((N, L), dtype=np.float32)
	X = np.zeros((batch, N), dtype=np.float32)
	A = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	for w in np.arange(W):
		M_w = w*batch
		if w == (W - 1): # Last batch
			X = np.zeros((M - M_w, N), dtype=np.float32)
		memory.memCenSVD(G, U0, V0, X, f, M_w) if d is None else memory.memFinSVD(G, U0, V0, X, f, d, M_w)
		H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for _ in np.arange(power):
		X = np.zeros((batch, N), dtype=np.float32)
		for w in np.arange(W):
			M_w = w*batch
			if w == (W - 1): # Last batch
				X = np.zeros((M - M_w, N), dtype=np.float32)
			memory.memCenSVD(G, U0, V0, X, f, M_w) if d is None else memory.memFinSVD(G, U0, V0, X, f, d, M_w)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
		H -= a*Q
		Q, S, _ = eigSVD(H)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	X = np.zeros((batch, N), dtype=np.float32)
	for w in np.arange(W):
		M_w = w*batch
		if w == (W - 1): # Last batch
			X = np.zeros((M - M_w, N), dtype=np.float32)
		memory.memCenSVD(G, U0, V0, X, f, M_w) if d is None else memory.memFinSVD(G, U0, V0, X, f, d, M_w)
		A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
	U, S, V = eigSVD(A)
	return np.ascontiguousarray(U[:,:K]), S[:K], np.ascontiguousarray(np.dot(Q, V)[:,:K])

# Batched randomized SVD with dynamic shift (frequency-based)
def memoryNoise(G, f, n, N, K, batch, power, rng):
	M = G.shape[0]
	W = ceil(M/batch)
	a = 0.0
	L = max(K + 10, 20)
	H = np.zeros((N, L), dtype=np.float32)
	X = np.zeros((batch, N), dtype=np.float32)
	A = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	for w in np.arange(W):
		M_w = w*batch
		if w == (W - 1): # Last batch
			X = np.zeros((M - M_w, N), dtype=np.float32)
		memory.memNoise(G, X, f, n, M_w)
		H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for _ in np.arange(power):
		X = np.zeros((batch, N), dtype=np.float32)
		for w in np.arange(W):
			M_w = w*batch
			if w == (W - 1): # Last batch
				X = np.zeros((M - M_w, N), dtype=np.float32)
			memory.memNoise(G, X, f, n, M_w)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
		H -= a*Q
		Q, S, _ = eigSVD(H)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	X = np.zeros((batch, N), dtype=np.float32)
	for w in np.arange(W):
		M_w = w*batch
		if w == (W - 1): # Last batch
			X = np.zeros((M - M_w, N), dtype=np.float32)
		memory.memNoise(G, X, f, n, M_w)
		A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
	U, S, V = eigSVD(A)
	return np.ascontiguousarray(U[:,:K]), S[:K], np.ascontiguousarray(np.dot(Q, V)[:,:K])


### EMU algorithm
def emuAlgorithm(G, E, f, N, e, K, run):
	if E is None:
		print("Using memory-efficient variant of EMU.")

	# Extract run options
	rng = np.random.default_rng(run["seed"])
	b = run["batch"]
	p = run["power"]

	# Estimate scaling vector
	d = 1.0/np.sqrt(2.0*f*(1.0 - f))

	# Exit without performing EMU
	if run["iter"] < 1:
		# Estimating final SVD (without iterations) 
		print("Warning, no EM-PCA iterations are performed!")
		print(f"Extracting {K} eigenvector(s).")
		if E is None:
			U, S, V = memoryInit(G, f, d, N, K, b, p, rng)
		else:
			shared.standardInit(G, E, f, d)
			U, S, V = randomizedSVD(E, K, p, rng)
		U, V = signFlip(U, V)

		# Create output structure
		res = {
			"U":U,
			"S":S,
			"V":V,
			"iter":0,
			"conv":False
		}
		return res
	else:
		# Estimate initial individual allele frequencies
		ts = time()
		print("Initiating accelerated scheme", end="")
		if run["noise"] is not None:
			n = rng.normal(0.0, run["noise"], size=G.shape[0]).astype(np.float32)
			if E is None:
				U, S, V = memoryNoise(G, f, n, N, e, b, p, rng)
			else:
				shared.centerNoise(G, E, f, n)
				U, S, V = randomizedSVD(E, e, p, rng)
			del n
		else:
			if E is None:
				U, S, V = memoryInit(G, f, None, N, e, b, p, rng)
			else:
				shared.centerInit(G, E, f)
				U, S, V = randomizedSVD(E, e, p, rng)
		U, V = signFlip(U, V)
		U *= np.sqrt(S)
		V *= np.sqrt(S)
		U_pre = np.copy(U)
		print(f"\rInitiating accelerated scheme\t({time() - ts:.1f}s)")

		# Iterative estimation of individual allele frequencies
		ts = time()
		for it in range(1, run["iter"] + 1):
			# 1st SVD step
			if E is None:
				U1, S1, V1 = memorySVD(G, U, V, f, None, N, e, b, p, rng)
			else:
				shared.centerAccel(G, E, U, V, f)
				U1, S1, V1 = randomizedSVD(E, e, p, rng)
			U1, V1 = signFlip(U1, V1)
			U1 *= np.sqrt(S1)
			V1 *= np.sqrt(S1)

			# 2nd SVD step
			if E is None:
				U2, S2, V2 = memorySVD(G, U1, V1, f, None, N, e, b, p, rng)
			else:
				shared.centerAccel(G, E, U1, V1, f)
				U2, S2, V2 = randomizedSVD(E, e, p, rng)
			U2, V2 = signFlip(U2, V2)
			U2 *= np.sqrt(S2)
			V2 *= np.sqrt(S2)

			# QN steps
			shared.alphaStep(U, U1, U2)
			shared.alphaStep(V, V1, V2)

			# Stabilization step
			if E is None:
				U, S, V = memorySVD(G, U, V, f, None, N, e, b, p, rng)
			else:
				shared.centerAccel(G, E, U, V, f)
				U, S, V = randomizedSVD(E, e, p, rng)
			U, V = signFlip(U, V)
			U *= np.sqrt(S)
			V *= np.sqrt(S)	

			# Break iterative update if converged
			rmseU = shared.rmse(U, U_pre)
			print(f"({it})\tRMSE = {rmseU:.7f}\t({time() - ts:.1f}s)")
			if rmseU < run["tole"]:
				print("EM-PCA has converged.")
				conv = True
				break
			if it == run["iter"]:
				print("EM-PCA did not converge!")
				conv = False
			memoryview(U_pre.ravel())[:] = memoryview(U.ravel())
			ts = time()
		del U1, U2, V1, V2, S1, S2, U_pre

		# Estimating final SVD
		print(f"Extracting {K} eigenvector(s).")
		if E is None:
			U, S, V = memorySVD(G, U, V, f, d, N, K, b, p, rng)
		else:
			shared.standardAccel(G, E, U, V, f, d)
			U, S, V = randomizedSVD(E, K, p, rng)
		U, V = signFlip(U, V)

		# Create output structure
		res = {
			"U":U,
			"S":S,
			"V":V,
			"iter":it,
			"conv":conv
		}
		return res
