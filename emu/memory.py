"""
EMU.
Iterative SVD algorithms.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from math import sqrt
from emu import shared
from emu import shared_cy

##### EMU-mem #####
def emuMemory(D, f, e, K, N, e_iter, e_tole, power, cost, batch, seed, threads):
	M, B = D.shape

	# Setup acceleration
	print("Using accelerated EM scheme (SqS3).")
	dU1 = np.zeros((M, e), dtype=np.float32)
	dU2 = np.zeros((M, e), dtype=np.float32)
	dU3 = np.zeros((M, e), dtype=np.float32)
	dV1 = np.zeros((N, e), dtype=np.float32)
	dV2 = np.zeros((N, e), dtype=np.float32)
	dV3 = np.zeros((N, e), dtype=np.float32)

	# Exit without performing EMU
	if e_iter < 1:
		print("Warning, no EM-PCA iterations are performed!")
		print("Inferring set of eigenvector(s).")
		U, S, V = shared.halkoBatch(D, f, K, N, None, None, None, power, batch, \
			seed, True, threads)
		return U, S, V, 0, False
	else:
		# Estimate initial individual allele frequencies
		print("Initiating accelerated EM scheme (1)")
		U, S, V = shared.halkoBatch(D, f, e, N, None, None, None, power, batch, \
			seed, False, threads)
		U, V = shared.signFlip(U, V)
		V = V*S
		U0 = np.zeros_like(U)
		V0 = np.zeros_like(V)
		seed += 1
		
		# Estimate cost
		if cost:
			sumV = np.zeros(M, dtype=np.float32)
			shared_cy.frobenius(D, f, U, V, sumV, threads)
			print(f"Frobenius: {np.round(np.sum(sumV, dtype=float),1)}")

		# Iterative estimation of individual allele frequencies
		for i in range(1, e_iter+1):
			np.copyto(U0, U, casting="no")
			np.copyto(V0, V, casting="no")

			# 1st SVD step
			U1, S1, V1 = shared.halkoBatch(D, f, e, N, U, None, V, power, batch, \
				seed, False, threads)
			U1, V1 = shared.signFlip(U1, V1)
			V1 = V1*S1
			seed += 1
			shared_cy.matMinus(U1, U, dU1)
			shared_cy.matMinus(V1, V, dV1)
			sr2_U = shared_cy.matSumSquare(dU1)
			sr2_V = shared_cy.matSumSquare(dV1)

			# 2nd SVD step
			U2, S2, V2 = shared.halkoBatch(D, f, e, N, U1, None, V1, power, batch, \
				seed, False, threads)
			U2, V2 = shared.signFlip(U2, V2)
			V2 = V2*S2
			seed += 1
			shared_cy.matMinus(U2, U1, dU2)
			shared_cy.matMinus(V2, V1, dV2)

			# SQUAREM update of V and U SqS3
			shared_cy.matMinus(dU2, dU1, dU3)
			shared_cy.matMinus(dV2, dV1, dV3)
			sv2_U = shared_cy.matSumSquare(dU3)
			sv2_V = shared_cy.matSumSquare(dV3)
			if i == 1:
				if np.isclose(sv2_U, 0.0):
					print("No missingness in data. Skipping iterative approach!")
					converged = False
					break
			aU = -max(1.0, sqrt(sr2_U/sv2_U))
			aV = -max(1.0, sqrt(sr2_V/sv2_V))

			# New accelerated update
			shared_cy.matUpdate(U, U0, dU1, dU3, aU)
			shared_cy.matUpdate(V, V0, dV1, dV3, aV)

			# Check optional cost function
			if cost:
				shared_cy.frobenius(D, f, U, V, sumV, threads)
				print(f"Frobenius: {np.round(np.sum(sumV, dtype=float),1)}")

			# Break iterative update if converged
			rmseU = shared_cy.rmse(U, U0)
			print(f"Iteration {i},\tRMSE={round(rmseU, 9)}")
			if rmseU < e_tole:
				print("EM-PCA has converged.")
				converged = True
				break
			if i == e_iter:
				print("EM-PCA did not converged!")
				converged = False
		del U0, U1, U2, V0, V1, V2, S1, S2, dU1, dU2, dU3, dV1, dV2, dV3

		# Stabilization step
		U, S, V = shared.halkoBatch(D, f, e, N, U, None, V, power, batch, seed, \
			False, threads)
		U, V = shared.signFlip(U, V)
		seed += 1

		# Estimating SVD
		print(f"Inferring {K} eigenvector(s).")
		U, S, V = shared.halkoBatch(D, f, K, N, U, S, V, power, batch, seed, \
			True, threads)
		U, V = shared.signFlip(U, V)
		return U, S, V, i, converged
