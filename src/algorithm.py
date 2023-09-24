"""
EMU.
Iterative SVD algorithms.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from math import sqrt
from src import shared
from src import shared_cy

##### EMU #####
def emuAlgorithm(D, f, e, K, N, e_iter, e_tole, power, cost, seed, threads):
	M, B = D.shape
	E = np.zeros((M, N), dtype=np.float32)

	# Setup acceleration
	print("Using accelerated EM scheme (SqS3).")
	DU1 = np.zeros((M, e), dtype=np.float32)
	DU2 = np.zeros((M, e), dtype=np.float32)
	DU3 = np.zeros((M, e), dtype=np.float32)
	DV1 = np.zeros((N, e), dtype=np.float32)
	DV2 = np.zeros((N, e), dtype=np.float32)
	DV3 = np.zeros((N, e), dtype=np.float32)

	# Initiate E matrix
	shared_cy.updateInit(D, f, E, threads)

	# Exit without performing EMU
	if e_iter < 1:
		print("Warning, no EM-PCA iterations are performed!")
		print("Inferring eigenvector(s).")
		shared_cy.standardizeMatrix(E, f, threads)
		U, S, V = shared.halko(E, K, power, seed)
		U, V = shared.signFlip(U, V)
		del E
		return U, S, V, 0, False
	else:
		# Estimate initial individual allele frequencies
		print("Initiating accelerated EM scheme (1)")
		U, S, V = shared.halko(E, e, power, seed)
		U, V = shared.signFlip(U, V)
		V = V*S
		seed += 1

		# Estimate cost
		if cost:
			sumV = np.zeros(M, dtype=np.float32)
			shared_cy.frobenius(D, f, U, V, sumV, threads)
			print(f"Frobenius: {np.round(np.sum(sumV, dtype=float),1)}")

		# Update E matrix
		shared_cy.updateAccel(D, E, f, U, V, threads)

		# Iterative estimation of individual allele frequencies
		for i in range(1, e_iter+1):
			U0 = np.copy(U)

			# 1st SVD step
			U1, S1, V1 = shared.halko(E, e, power, seed)
			U1, V1 = shared.signFlip(U1, V1)
			V1 = V1*S1
			seed += 1
			shared_cy.matMinus(U1, U, DU1)
			shared_cy.matMinus(V1, V, DV1)
			sr2_U = shared_cy.matSumSquare(DU1)
			sr2_V = shared_cy.matSumSquare(DV1)
			shared_cy.updateAccel(D, E, f, U1, V1, threads)

			# 2nd SVD step
			U2, S2, V2 = shared.halko(E, e, power, seed)
			U2, V2 = shared.signFlip(U2, V2)
			V2 = V2*S2
			seed += 1
			shared_cy.matMinus(U2, U1, DU2)
			shared_cy.matMinus(V2, V1, DV2)

			# SQUAREM update of U and V SqS3
			shared_cy.matMinus(DU2, DU1, DU3)
			shared_cy.matMinus(DV2, DV1, DV3)
			sv2_U = shared_cy.matSumSquare(DU3)
			sv2_V = shared_cy.matSumSquare(DV3)
			if i == 1:
				if np.isclose(sv2_U, 0.0):
					print("No missingness in data. Skipping iterative approach!")
					converged = False
					break
			aU = max(1.0, sqrt(sr2_U/sv2_U))
			aV = max(1.0, sqrt(sr2_V/sv2_V))

			# Accelerated update
			shared_cy.matUpdate(U, DU1, DU3, aU)
			shared_cy.matUpdate(V, DV1, DV3, aV)
			shared_cy.updateAccel(D, E, f, U, V, threads)

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
		del U0, U1, U2, V1, V2, S1, S2, DU1, DU2, DU3, DV1, DV2, DV3

		# Stabilization step
		U, S, V = shared.halko(E, e, power, seed)
		U, V = shared.signFlip(U, V)
		seed += 1
		shared_cy.updateSVD(D, E, f, U, S, V, threads)

		# Estimating SVD
		shared_cy.standardizeMatrix(E, f, threads)
		print(f"Inferring {K} eigenvector(s).")
		U, S, V = shared.halko(E, K, power, seed)
		U, V = shared.signFlip(U, V)
		del E
		return U, S, V, i, converged
