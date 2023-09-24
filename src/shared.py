"""
EMU.
Iterative SVD algorithms.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import subprocess
import numpy as np
from math import ceil
from src import memory_cy

##### Shared functions #####
### Helper functions
# Reader help function
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, err = process.communicate()
	return int(result.split()[0])

# Flip signs of SVD output - Based on scikit-learn (svd_flip)
def signFlip(U, V):
    maxCols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[maxCols, range(U.shape[1])])
    U *= signs
    V *= signs
    return U, V

### Randomized SVD functions
# PCAone Halko full
def halko(E, K, power, seed):
	M, N = E.shape
	L = K + 16
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, L)).astype(np.float32)
	A = np.zeros((M, L), dtype=np.float32)
	H = np.zeros((N, L), dtype=np.float32)
	for p in range(power):
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
		np.dot(E, O, out=A)
		np.dot(E.T, A, out=H)
	Q, R = np.linalg.qr(A, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, Uhat
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, S[:K], V

# PCAone Halko acceleration
def halkoBatch(D, f, K, N, U0, S0, V0, power, batch, seed, final, threads):
	M, B = D.shape
	W = ceil(M/batch)
	L = K + 16
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
			if U0 is None:
				if final:
					memory_cy.plinkFinalFreq(D, E, f, M_w, threads)
				else:
					memory_cy.plinkFreq(D, E, f, M_w, threads)
			else:
				if final:
					memory_cy.plinkFinalSVD(D, E, U0, S0, V0, f, M_w, threads)
				else:
					memory_cy.plinkSVD(D, E, U0, V0, f, M_w, threads)
			A[M_w:(M_w + E.shape[0])] = np.dot(E, O)
			H += np.dot(E.T, A[M_w:(M_w + E.shape[0])])
	Q, R = np.linalg.qr(A, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, Uhat, E
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, S[:K], V
