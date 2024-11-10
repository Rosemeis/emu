"""
EMU.
Iterative SVD algorithms for genetic data with missingness.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from math import ceil
from emu import memory

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
# PCAone Halko full
def halko(E, K, power, seed):
	M, N = E.shape
	L = K + 20
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, L)).astype(np.float32)
	A = np.zeros((M, L), dtype=np.float32)
	H = np.zeros((N, L), dtype=np.float32)
	for p in range(power):
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
		np.dot(E, O, out=A)
		np.dot(E.T, A, out=H)
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, R1, R2, Uhat
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, S[:K], V

# Batch PCAone Halko - Frequency
def halkoBatchFreq(G, f, d, K, N, power, batch, seed, final, threads):
	M = G.shape[0]
	W = ceil(M/batch)
	L = K + 20
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, L)).astype(np.float32)
	A = np.zeros((M, L), dtype=np.float32)
	H = np.zeros((N, L), dtype=np.float32)
	for p in range(power):
		E = np.zeros((batch, N), dtype=np.float32)
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for w in range(W):
			M_w = w*batch
			if w == (W-1): # Last batch
				del E # Ensure no extra copy
				E = np.zeros((M - M_w, N), dtype=np.float32)
			if final:
				memory.plinkFinalFreq(G, E, f, d, M_w, threads)
			else:
				memory.plinkFreq(G, E, f, M_w, threads)
			A[M_w:(M_w + E.shape[0])] = np.dot(E, O)
			H += np.dot(E.T, A[M_w:(M_w + E.shape[0])])
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, R1, R2, Uhat, E
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, S[:K], V

# Batch PCAone Halko - SVD
def halkoBatchSVD(G, f, d, K, N, U0, S0, V0, power, batch, seed, final, threads):
	M = G.shape[0]
	W = ceil(M/batch)
	L = K + 20
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, L)).astype(np.float32)
	A = np.zeros((M, L), dtype=np.float32)
	H = np.zeros((N, L), dtype=np.float32)
	for p in range(power):
		E = np.zeros((batch, N), dtype=np.float32)
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for w in range(W):
			M_w = w*batch
			if w == (W-1): # Last batch
				del E # Ensure no extra copy
				E = np.zeros((M - M_w, N), dtype=np.float32)
			if final:
				memory.plinkFinalSVD(G, E, U0, S0, V0, f, d, M_w, threads)
			else:
				memory.plinkSVD(G, E, U0, V0, f, M_w, threads)
			A[M_w:(M_w + E.shape[0])] = np.dot(E, O)
			H += np.dot(E.T, A[M_w:(M_w + E.shape[0])])
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, R1, R2, Uhat, E
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, S[:K], V
