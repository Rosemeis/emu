"""
EMU.
Iterative SVD algorithms for genetic data with missingness.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from math import sqrt
from emu import functions
from emu import shared

##### EMU #####
# Standard variant of EMU
def emuAlg(G, f, e, K, N, iter, tole, power, cost, seed, threads):
	M = G.shape[0]
	E = np.zeros((M, N), dtype=np.float32)
	d = 1.0/np.sqrt(2.0*f*(1-f))

	# Setup acceleration
	print("Using accelerated EM scheme (QN).")
	U1 = np.zeros((M, e), dtype=np.float32)
	U2 = np.zeros((M, e), dtype=np.float32)
	V1 = np.zeros((N, e), dtype=np.float32)
	V2 = np.zeros((N, e), dtype=np.float32)

	# Initiate E matrix
	shared.updateInit(G, f, E, threads)

	# Exit without performing EMU
	if iter < 1:
		print("Warning, no EM-PCA iterations are performed!")
		print("Inferring eigenvector(s).")
		shared.standardizeMatrix(E, d, threads)
		U, S, V = functions.halko(E, K, power, seed)
		U, V = functions.signFlip(U, V)
		del E, d
		return U, S, V, 0, False
	else:
		# Estimate initial individual allele frequencies
		print("Initiating accelerated EM scheme (1)")
		U, S, V = functions.halko(E, e, power, seed)
		U, V = functions.signFlip(U, V)
		V *= S
		U_old = np.zeros_like(U)
		V_old = np.zeros_like(V)

		# Estimate cost
		if cost:
			sumV = np.zeros(M, dtype=np.float32)
			shared.frobenius(G, f, U, V, sumV, threads)
			print(f"Frobenius: {np.round(np.sum(sumV, dtype=float),1)}")

		# Update E matrix
		shared.updateAccel(G, E, f, U, V, threads)

		# Iterative estimation of individual allele frequencies
		for it in range(1, iter+1):
			memoryview(U_old.ravel())[:] = memoryview(U.ravel())
			memoryview(V_old.ravel())[:] = memoryview(V.ravel())

			# 1st SVD step
			seed += 1
			U1, S1, V1 = functions.halko(E, e, power, seed)
			U1, V1 = functions.signFlip(U1, V1)
			V1 *= S1
			shared.updateAccel(G, E, f, U1, V1, threads)

			# 2nd SVD step
			seed += 1
			U2, S2, V2 = functions.halko(E, e, power, seed)
			U2, V2 = functions.signFlip(U2, V2)
			V2 *= S2

			# QN steps
			shared.alphaStep(U, U1, U2)
			shared.alphaStep(V, V1, V2)
			shared.updateAccel(G, E, f, U, V, threads)

			# Check optional cost function
			if cost:
				shared.frobenius(G, f, U, V, sumV, threads)
				print(f"Frobenius: {np.round(np.sum(sumV, dtype=float),1)}")

			# Break iterative update if converged
			rmseU = shared.rmse(U, U_old)
			print(f"Iteration {it},\tRMSE={round(rmseU, 9)}")
			if rmseU < tole:
				print("EM-PCA has converged.")
				converged = True
				break
			if it == iter:
				print("EM-PCA did not converge!")
				converged = False
		del U1, U2, U_old, V1, V2, V_old, S1, S2

		# Stabilization step
		seed += 1
		U, S, V = functions.halko(E, e, power, seed)
		U, V = functions.signFlip(U, V)
		shared.updateSVD(G, E, f, U, S, V, threads)

		# Estimating SVD
		shared.standardizeMatrix(E, d, threads)
		print(f"Inferring {K} eigenvector(s).")
		seed += 1
		U, S, V = functions.halko(E, K, power, seed)
		U, V = functions.signFlip(U, V)
		del E, d
		return U, S, V, it, converged


# Memory-efficient variant
def emuMem(G, f, e, K, N, iter, tole, power, cost, batch, seed, threads):
	M = G.shape[0]
	d = 1.0/np.sqrt(2.0*f*(1-f))

	# Setup acceleration
	print("Using accelerated EM scheme (QN).")
	U1 = np.zeros((M, e), dtype=np.float32)
	U2 = np.zeros((M, e), dtype=np.float32)
	V1 = np.zeros((N, e), dtype=np.float32)
	V2 = np.zeros((N, e), dtype=np.float32)

	# Exit without performing EMU
	if iter < 1:
		print("Warning, no EM-PCA iterations are performed!")
		print("Inferring set of eigenvector(s).")
		U, S, V = functions.halkoBatchFreq(G, f, d, K, N, power, batch, \
			seed, True, threads)
		del d
		return U, S, V, 0, False
	else:
		# Estimate initial individual allele frequencies
		print("Initiating accelerated EM scheme (1)")
		U, S, V = functions.halkoBatchFreq(G, f, d, e, N, power, batch, \
			seed, False, threads)
		U, V = functions.signFlip(U, V)
		V *= S
		U_old = np.zeros_like(U)
		V_old = np.zeros_like(V)
		
		# Estimate cost
		if cost:
			sumV = np.zeros(M, dtype=np.float32)
			shared.frobenius(G, f, U, V, sumV, threads)
			print(f"Frobenius: {np.round(np.sum(sumV, dtype=float),1)}")

		# Iterative estimation of individual allele frequencies
		for it in range(1, iter+1):
			memoryview(U_old.ravel())[:] = memoryview(U.ravel())
			memoryview(V_old.ravel())[:] = memoryview(V.ravel())

			# 1st SVD step
			seed += 1
			U1, S1, V1 = functions.halkoBatchSVD(G, f, d, e, N, U, None, V, power, \
				batch, seed, False, threads)
			U1, V1 = functions.signFlip(U1, V1)
			V1 *= S1

			# 2nd SVD step
			seed += 1
			U2, S2, V2 = functions.halkoBatchSVD(G, f, d, e, N, U1, None, V1, power, \
				batch, seed, False, threads)
			U2, V2 = functions.signFlip(U2, V2)
			V2 *= S2

			# QN steps
			shared.alphaStep(U, U1, U2)
			shared.alphaStep(V, V1, V2)

			# Check optional cost function
			if cost:
				shared.frobenius(G, f, U, V, sumV, threads)
				print(f"Frobenius: {np.round(np.sum(sumV, dtype=float),1)}")

			# Break iterative update if converged
			rmseU = shared.rmse(U, U_old)
			print(f"Iteration {it},\tRMSE={round(rmseU, 9)}")
			if rmseU < tole:
				print("EM-PCA has converged.")
				converged = True
				break
			if it == iter:
				print("EM-PCA did not converged!")
				converged = False
		del U_old, U1, U2, V_old, V1, V2, S1, S2

		# Stabilization step
		seed += 1
		U, S, V = functions.halkoBatchSVD(G, f, d, e, N, U, None, V, power, batch, \
			seed, False, threads)
		U, V = functions.signFlip(U, V)

		# Estimating SVD
		print(f"Inferring {K} eigenvector(s).")
		seed += 1
		U, S, V = functions.halkoBatchSVD(G, f, d, K, N, U, S, V, power, batch, \
			seed, True, threads)
		U, V = functions.signFlip(U, V)
		del d
		return U, S, V, it, converged
