"""
EMU.
Iterative SVD algorithms.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds

# Import own scripts
from emu import halko
from emu import shared_cy

##### EMU #####
# Flip signs of SVD output - Based on scikit-learn (svd_flip)
def signFlip(U, Vt):
    maxCols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[maxCols, range(U.shape[1])])
    U *= signs
    Vt *= signs[:, np.newaxis]
    return U, Vt

### PCAone Halko ###
def halkoPCAone(E, k, n_iter, random_state):
	m, n = E.shape
	Omg = np.random.standard_normal(size=(n, k+10)).astype(np.float32)
	for p in range(n_iter):
		if p > 0:
			Omg, _ = np.linalg.qr(H, mode="reduced")
		G = np.dot(E, Omg)
		H = np.dot(E.T, G)
	Q, R = np.linalg.qr(G, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, s, V = np.linalg.svd(B, full_matrices=False)
	del B
	U = np.dot(Q, Uhat)
	return U[:,:k], s[:k], V[:k,:]


### Main EMU function ###
def emuAlgorithm(D, f, e, K, M, M_tole, Bi, n, m, svd_method, svd_power, \
					output, accel, cost, cost_step, seed, t):
	E = np.zeros((m, n), dtype=np.float32)

	# Setup acceleration
	if accel:
		print("Using accelerated EM scheme (SqS3)")
		diffU_1 = np.zeros((m, e), dtype=np.float32)
		diffU_2 = np.zeros((m, e), dtype=np.float32)
		diffU_3 = np.zeros((m, e), dtype=np.float32)
		diffW_1 = np.zeros((e, n), dtype=np.float32)
		diffW_2 = np.zeros((e, n), dtype=np.float32)
		diffW_3 = np.zeros((e, n), dtype=np.float32)

	# Initiate E matrix
	shared_cy.updateE_init(D, f, E, Bi, t)

	# Exit without performing EMU
	if M < 1:
		print("Warning, no EM-PCA iterations are performed!")
		# Estimating SVD
		shared_cy.standardizeMatrix(E, f, t)
		print("Inferring set of eigenvector(s).")
		if svd_method == "arpack":
			U, s, V = svds(E, k=K)
			U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
			U, V = signFlip(U, V)
		elif svd_method == "halko":
			U, s, V = halkoPCAone(E, K, svd_power, seed)
			U, V = signFlip(U, V)
		del E
		return U, s, V
	else:
		if accel:
			print("Initiating accelerated EM scheme (1).")
		# Estimate initial individual allele frequencies
		if svd_method == "arpack":
			U, s, W = svds(E, k=e)
			U, W = signFlip(U, W)
		elif svd_method == "halko":
			U, s, W = halkoPCAone(E, K, svd_power, seed)
			U, W = signFlip(U, W)

		# Estimate cost
		if cost:
			sumVec = np.zeros(m, dtype=np.float32)
			shared_cy.frobenius(D, f, U, s, W, sumVec, Bi, t)
			oldCost = np.sum(sumVec)
			print("Frobenius: " + str(oldCost))

		# Update E matrix based on setting
		if not accel:
			shared_cy.updateE_SVD(D, E, f, U, s, W, Bi, t)
			print("Individual allele frequencies estimated (1).")
		else:
			W = W*s.reshape((e, 1))
			shared_cy.updateE_SVD_accel(D, E, f, U, W, Bi, t)

		# Iterative estimation of individual allele frequencies
		for i in range(M):
			prevU = np.copy(U)
			if accel:
				if svd_method == "arpack":
					U1, s1, W1 = svds(E, k=e)
					U1, W1 = signFlip(U1, W1)
				elif svd_method == "halko":
					U1, s1, W1 = halkoPCAone(E, K, svd_power, seed)
					U1, W1 = signFlip(U1, W1)
				W1 = W1*s1.reshape((e, 1))
				shared_cy.matMinus(U1, U, diffU_1)
				shared_cy.matMinus(W1, W, diffW_1)
				sr2_U = shared_cy.matSumSquare(diffU_1)
				sr2_W = shared_cy.matSumSquare(diffW_1)
				shared_cy.updateE_SVD_accel(D, E, f, U1, W1, Bi, t)
				if svd_method == "arpack":
					U2, s2, W2 = svds(E, k=e)
					U2, W2 = signFlip(U2, W2)
				elif svd_method == "halko":
					U2, s2, W2 = halkoPCAone(E, K, svd_power, seed)
					U2, W2 = signFlip(U2, W2)
				W2 = W2*s2.reshape((e, 1))
				shared_cy.matMinus(U2, U1, diffU_2)
				shared_cy.matMinus(W2, W1, diffW_2)

				# SQUAREM update of W and U SqS3
				shared_cy.matMinus(diffU_2, diffU_1, diffU_3)
				shared_cy.matMinus(diffW_2, diffW_1, diffW_3)
				sv2_U = shared_cy.matSumSquare(diffU_3)
				sv2_W = shared_cy.matSumSquare(diffW_3)
				if (i == 0) and (np.isclose(sv2_U, 0.0)):
					print("No missingness in data. Skipping iterative approach!")
					break
				alpha_U = max(1.0, np.sqrt(sr2_U/sv2_U))
				alpha_W = max(1.0, np.sqrt(sr2_W/sv2_W))

				# New accelerated update
				shared_cy.matUpdate(U, diffU_1, diffU_3, alpha_U)
				shared_cy.matUpdate(W, diffW_1, diffW_3, alpha_W)
				shared_cy.updateE_SVD_accel(D, E, f, U, W, Bi, t)
				if cost:
					shared_cy.frobenius_accel(D, f, U, W, sumVec, Bi, t)
					newCost = np.sum(sumVec)
					print("Frobenius: " + str(newCost))
					if oldCost >= newCost:
						print("Bad step, using un-accelerated update!")
						shared_cy.updateE_SVD_accel(D, E, f, U2, W2, Bi, t)
					else:
						oldCost = newCost
			else:
				if svd_method == "arpack":
					U, s, W = svds(E, k=e)
					U, W = signFlip(U, W)
				elif svd_method == "halko":
					U, s, W = halkoPCAone(E, K, svd_power, seed)
					U, W = signFlip(U, W)
				shared_cy.updateE_SVD(D, E, f, U, s, W, Bi, t)
				if cost:
					shared_cy.frobenius(D, f, U, s, W, sumVec, Bi, t)
					print("Frobenius: " + str(np.sum(sumVec)))

			# Break iterative update if converged
			diff = np.sqrt(np.sum(shared_cy.rmse(U, prevU))/(m*e))
			print("Individual allele frequencies estimated (" + str(i+2) + "). RMSE=" + str(diff))
			if diff < M_tole:
				print("Estimation of individual allele frequencies has converged.")
				break
		del prevU
		if cost:
			del sumVec

		# Run non-accelerated update to ensure properties of W, s, U
		if accel:
			if svd_method == "arpack":
				U, s, W = svds(E, k=e)
				U, W = signFlip(U, W)
			elif svd_method == "halko":
				U, s, W = halkoPCAone(E, K, svd_power, seed)
				U, W = signFlip(U, W)
			shared_cy.updateE_SVD(D, E, f, U, s, W, Bi, t)
			del U1, U2, W1, W2, s1, s2, diffU_1, diffU_2, diffU_3, diffW_1, diffW_2, diffW_3

		# Estimating SVD
		shared_cy.standardizeMatrix(E, f, t)
		print("Inferring set of eigenvector(s).")
		if svd_method == "arpack":
			U, s, V = svds(E, k=K)
			U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
		elif svd_method == "halko":
			U, s, V = halkoPCAone(E, K, svd_power, seed)
			U, W = signFlip(U, W)
		del E
		return U, s, V

##### EMU-mem ######
### Range finder functions of Q
def range_finder(D, f, e, Bi, n, m, svd_power, t):
	K = e + 10
	C = np.zeros((m, K), dtype=np.float32)

	# Sample Gaussian vectors
	np.random.seed(0)
	Q = np.random.normal(size=(n, K)).astype(np.float32, copy=False)

	# Power iterations
	for pow_i in range(svd_power):
		C.fill(0)
		halko.matMul_Freq(D, f, Q, C, Bi, n, m, t)
		Q.fill(0)
		halko.matMulTrans_Freq(D, f, C, Q, Bi, n, m, t)
	C.fill(0)
	halko.matMul_Freq(D, f, Q, C, Bi, n, m, t)
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
def range_finder_final(D, f, e, U, s, W, Bi, n, m, svd_power, t):
	K = e + 10
	C = np.zeros((m, K), dtype=np.float32)

	# Sample Gaussian vectors
	np.random.seed(0)
	Q = np.random.normal(size=(n, K)).astype(np.float32, copy=False)

	# Power iterations
	for pow_i in range(svd_power):
		if W is None:
			C.fill(0)
			halko.matMulFinal_Freq(D, f, Q, C, Bi, n, m, t)
			C, _ = linalg.lu(C, permute_l=True)
			Q = np.zeros((n, K), dtype=np.float32)
			halko.matMulTransFinal_Freq(D, f, C, Q, Bi, n, m, t)
			Q, _ = linalg.lu(Q, permute_l=True)
		else:
			C.fill(0)
			halko.matMulFinal_SVD(D, f, U, s, W, Q, C, Bi, n, m, t)
			C, _ = linalg.lu(C, permute_l=True)
			Q = np.zeros((n, K), dtype=np.float32)
			halko.matMulTransFinal_SVD(D, f, U, s, W, C, Q, Bi, n, m, t)
			Q, _ = linalg.lu(Q, permute_l=True)
	C.fill(0)
	if W is None:
		halko.matMulFinal_Freq(D, f, Q, C, Bi, n, m, t)
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
def customSVD(D, f, e, Bi, n, m, svd_power, t):
	Q = range_finder(D, f, e, Bi, n, m, svd_power, t)
	Bt = np.zeros((n, Q.shape[1]), dtype=np.float32)

	# B.T = dot(E.T, Q)
	halko.matMulTrans_Freq(D, f, Q, Bt, Bi, n, m, t)

	# SVD on thin matrix
	Uhat, s, V = linalg.svd(Bt.T, full_matrices=False)
	del Bt
	U = np.dot(Q, Uhat)

	# Correct sign
	U, V = signFlip(U, V)
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
	U, V = signFlip(U, V)
	return U[:,:e], s[:e], V[:e,:]

# Final SVD
def customFinalSVD(D, f, e, U, s, W, Bi, n, m, svd_power, t):
	Q = range_finder_final(D, f, e, U, s, W, Bi, n, m, svd_power, t)
	Bt = np.zeros((n, Q.shape[1]), dtype=np.float32)

	# B.T = dot(E.T, Q)
	if W is None:
		halko.matMulTransFinal_Freq(D, f, Q, Bt, Bi, n, m, t)
	else:
		halko.matMulTransFinal_SVD(D, f, U, s, W, Q, Bt, Bi, n, m, t)

	# SVD on thin matrix
	Uhat, s, V = linalg.svd(Bt.T, full_matrices=False)
	del Bt
	U = np.dot(Q, Uhat)

	# Correct sign
	U, V = signFlip(U, V)
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
	U, V = signFlip(U, V)
	return U[:,:e], s[:e], V[:e,:]


### Main EMU-mem function ###
def emuMemory(D, f, e, K, M, M_tole, Bi, n, m, svd_power, \
				output, accel, t):
	# Setup acceleration
	if accel:
		print("Using accelerated EM scheme (SqS3).")
		diffU_1 = np.zeros((m, e), dtype=np.float32)
		diffU_2 = np.zeros((m, e), dtype=np.float32)
		diffU_3 = np.zeros((m, e), dtype=np.float32)
		diffW_1 = np.zeros((e, n), dtype=np.float32)
		diffW_2 = np.zeros((e, n), dtype=np.float32)
		diffW_3 = np.zeros((e, n), dtype=np.float32)
	if M < 1:
		print("Warning, no EM-PCA iterations are performed!")
		print("Inferring set of eigenvector(s).")
		U, s, V = customFinalSVD(D, f, e, None, None, None, Bi, n, m, svd_power, t)
		return U, s, V
	else:
		# Estimate initial individual allele frequencies
		if accel:
			print("Initiating accelerated EM scheme (1)")
		U, s, W = customSVD(D, f, e, Bi, n, m, svd_power, t)
		if not accel:
			print("Individual allele frequencies estimated (1).")
		else:
			W = W*s.reshape((e, 1))

		# Iterative estimation of individual allele frequencies
		for iteration in range(2, M+1):
			prevU = np.copy(U)
			if accel:
				U1, s1, W1 = customDomainSVD_accel(D, f, e, U, W, Bi, n, m, svd_power, t)
				W1 = W1*s1.reshape((e, 1))
				shared_cy.matMinus(U1, U, diffU_1)
				shared_cy.matMinus(W1, W, diffW_1)
				sr2_U = shared_cy.matSumSquare(diffU_1)
				sr2_W = shared_cy.matSumSquare(diffW_1)
				U2, s2, W2 = customDomainSVD_accel(D, f, e, U1, W1, Bi, n, m, svd_power, t)
				W2 = W2*s2.reshape((e, 1))
				shared_cy.matMinus(U2, U1, diffU_2)
				shared_cy.matMinus(W2, W1, diffW_2)

				# SQUAREM update of W and U SqS3
				shared_cy.matMinus(diffU_2, diffU_1, diffU_3)
				shared_cy.matMinus(diffW_2, diffW_1, diffW_3)
				sv2_U = shared_cy.matSumSquare(diffU_3)
				sv2_W = shared_cy.matSumSquare(diffW_3)
				alpha_U = max(1.0, np.sqrt(sr2_U/sv2_U))
				alpha_W = max(1.0, np.sqrt(sr2_W/sv2_W))

				# New accelerated update
				shared_cy.matUpdate(U, diffU_1, diffU_3, alpha_U)
				shared_cy.matUpdate(W, diffW_1, diffW_3, alpha_W)
			else:
				U, s, W = customDomainSVD(D, f, e, U, s, W, Bi, n, m, svd_power, t)

			# Break iterative update if converged
			diff = np.sqrt(np.sum(shared_cy.rmse(U, prevU))/(m*e))
			print("Individual allele frequencies estimated (" + str(iteration) + "). RMSE=" + str(diff))
			if diff < M_tole:
				print("Estimation of individual allele frequencies has converged.")
				break
		del prevU

		# Run non-accelerated update to ensure properties of U, s, W
		if accel:
			U, s, W = customDomainSVD_accel(D, f, e, U, W, Bi, n, m, svd_power, t)
			del U1, U2, s1, s2, W1, W2, diffU_1, diffU_2, diffU_3, diffW_1, diffW_2, diffW_3

		# Estimating SVD
		print("Inferring set of eigenvector(s).")
		U, s, V = customFinalSVD(D, f, K, U, s, W, Bi, n, m, svd_power, t)
		return U, s, V
