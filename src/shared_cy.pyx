# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### EMU #####
# Estimate population allele frequencies
cpdef void estimateF(unsigned char[:,::1] D, float[::1] f, int N, int t) nogil:
	cdef:
		int M = D.shape[0]
		int B = D.shape[1]
		int i, j, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
		float n
	for j in prange(M, num_threads=t):
		i = 0
		n = 0.0
		for b in range(B):
			byte = D[j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					f[j] += <float>recode[byte & mask]
					n = n + 1.0
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break
		if n > 0.0:
			f[j] /= (2.0*n)
		else:
			f[j] = 0.0

# Array filtering
cpdef void filterArrays(unsigned char[:,::1] D, float[::1] f, \
		unsigned char[::1] mask) nogil:
	cdef:
		int M = D.shape[0]
		int B = D.shape[1]
		int c = 0
		int j, b
	for j in range(M):
		if mask[j] == 1:
			for b in range(B):
				D[c,b] = D[j,b]
			f[c] = f[j]
			c = c + 1

# Initial update of dosage matrix
cpdef void updateInit(unsigned char[:,::1] D, float[::1] f, float[:,::1] E, \
		int t) nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int B = D.shape[1]
		int i, j, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = D[j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					E[j,i] = <float>recode[byte & mask] - 2.0*f[j]
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Update E directly from SVD
cpdef void updateSVD(unsigned char[:,::1] D, float[:,::1] E, float[::1] f, \
		float[:,::1] U, float[::1] S, float[:,::1] V, int t) nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int B = D.shape[1]
		int K = U.shape[1]
		int i, j, k, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = D[j,b]
			for bytepart in range(4):
				if recode[byte & mask] == 9:
					E[j,i] = 0.0
					for k in range(K):
						E[j,i] += U[j,k]*S[k]*V[i,k]
					E[j,i] = min(max(E[j,i] + 2.0*f[j], 1e-4), 2-(1e-4))
					E[j,i] -= 2.0*f[j]
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Accelerated update of E directly from SVD
cpdef void updateAccel(unsigned char[:,::1] D, float[:,::1] E, float[::1] f, \
		float[:,:] U, float[:,:] V, int t) nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int B = D.shape[1]
		int K = U.shape[1]
		int i, j, k, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = D[j,b]
			for bytepart in range(4):
				if recode[byte & mask] == 9:
					E[j,i] = 0.0
					for k in range(K):
						E[j,i] += U[j,k]*V[i,k]
					E[j,i] = min(max(E[j,i] + 2.0*f[j], 1e-4), 2-(1e-4))
					E[j,i] -= 2.0*f[j]
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Standardize dosage matrix
cpdef void standardizeMatrix(float[:,::1] E, float[::1] f, int t) nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int i, j
	for j in prange(M, num_threads=t):
		for i in range(N):
			E[j,i] /= sqrt(2.0*f[j]*(1.0 - f[j]))

# Root-mean squared error
cpdef float rmse(float[:,::1] A, float[:,::1] B) nogil:
	cdef:
		int M = A.shape[0]
		int N = A.shape[1]
		int i, j
		float res = 0.0
	for j in range(M):
		for i in range(N):
			res += (A[j,i] - B[j,i])*(A[j,i] - B[j,i])
	return sqrt(res/(<float>(M*N)))

# Selection scan
cpdef void galinskyScan(float[:,::1] U, float[:,::1] Dsquared) nogil:
	cdef:
		int M = U.shape[0]
		int K = U.shape[1]
		int j, k
	for j in range(M):
		for k in range(K):
			Dsquared[j,k] = (U[j,k]**2)*float(M)

### Accelerated EM
# Matrix subtraction
cpdef void matMinus(float[:,::1] A, float[:,::1] B, float[:,::1] C) nogil:
	cdef:
		int M = A.shape[0]
		int N = A.shape[1]
		int i, j
	for j in range(M):
		for i in range(N):
			C[j,i] = A[j,i] - B[j,i]

# Matrix sum of squares
cpdef float matSumSquare(float[:,::1] A) nogil:
	cdef:
		int M = A.shape[0]
		int N = A.shape[1]
		int i, j
		float res = 0.0
	for j in range(M):
		for i in range(N):
			res += A[j,i]*A[j,i]
	return res

# Update factor matrices
cpdef void matUpdate(float[:,::1] A, float[:,::1] D1, float[:,::1] D3, \
		float alpha) nogil:
	cdef:
		int M = A.shape[0]
		int N = A.shape[1]
		int i, j
	for j in range(M):
		for i in range(N):
			A[j,i] = A[j,i] + 2*alpha*D1[j,i] + alpha*alpha*D3[j,i]

### Likelihood measures for debugging
cpdef void frobenius(unsigned char[:,::1] D, float[::1] f, float[:,::1] U, \
		float[:,::1] V, float[::1] sumV, int t) nogil:
	cdef:
		int M = U.shape[0]
		int K = U.shape[1]
		int N = V.shape[0]
		int B = D.shape[1]
		int i, j, k, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
		float e
	for j in prange(M, num_threads=t):
		i = 0
		sumV[j] = 0.0
		for b in range(B):
			byte = D[j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					e = 0.0
					for k in range(K):
						e = e + U[j,k]*V[i,k]
					e = min(max(e + 2.0*f[j], 1e-4), 2-(1e-4))
					sumV[j] += (<float>recode[byte & mask] - e)**2
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break
