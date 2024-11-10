# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### EMU #####
# Inline functions
cdef inline float project(const float e, const float f) noexcept nogil:
	return min(max(e + 2.0*f, 1e-4), 2.0-(1e-4))

cdef inline float innerE(const float* u, const float* s, const float* v, \
		const float f, const int K) noexcept nogil:
	cdef:
		int k
		float e = 0.0
	for k in range(K):
		e += u[k]*s[k]*v[k]
	return project(e, f) - 2.0*f

cdef inline float innerAccelE(const float* u, const float* v, const float f, \
		const int K) noexcept nogil:
	cdef:
		int k
		float e = 0.0
	for k in range(K):
		e += u[k]*v[k]
	return project(e, f) - 2.0*f

cdef inline float computeC(const float* x0, const float* x1, const float* x2, \
		const int I, const int J) noexcept nogil:
	cdef:
		int i, j, k
		float sum1 = 0.0
		float sum2 = 0.0
		float u, v
	for i in range(I):
		for j in range(J):
			k = i*J + j
			u = x1[k]-x0[k]
			v = (x2[k]-x1[k])-u
			sum1 += u*u
			sum2 += u*v
	return -(sum1/sum2)

# Estimate population allele frequencies
cpdef void estimateF(const unsigned char[:,::1] G, float[::1] f, int[::1] n, \
		const int N, const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					f[j] += <float>g
					n[j] += 1
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break
		if n[j] > 0:
			f[j] /= <float>(2*n[j])
		else:
			f[j] = 0.0

# Initial update of dosage matrix
cpdef void updateInit(const unsigned char[:,::1] G, const float[::1] f, \
		float[:,::1] E, const int t) noexcept nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int B = G.shape[1]
		int b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					E[j,i] = <float>g - 2.0*f[j]
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Update E directly from SVD
cpdef void updateSVD(const unsigned char[:,::1] G, float[:,::1] E, \
		const float[::1] f, const float[:,::1] U, const float[::1] S, \
		const float[:,::1] V, const int t) noexcept nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int B = G.shape[1]
		int K = U.shape[1]
		int b, i, j, k, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				if recode[byte & mask] == 9:
					E[j,i] = innerE(&U[j,0], &S[0], &V[i,0], f[j], K)
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Accelerated update of E directly from SVD
cpdef void updateAccel(const unsigned char[:,::1] G, float[:,::1] E, \
		const float[::1] f, const float[:,:] U, const float[:,:] V, \
		const int t) noexcept nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int B = G.shape[1]
		int K = U.shape[1]
		int b, i, j, k, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				if recode[byte & mask] == 9:
					E[j,i] = innerAccelE(&U[j,0], &V[i,0], f[j], K)
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Standardize dosage matrix
cpdef void standardizeMatrix(float[:,::1] E, const float[::1] d, const int t) \
		noexcept nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int i, j
	for j in prange(M, num_threads=t):
		for i in range(N):
			E[j,i] *= d[j]

# Root-mean squared error
cpdef float rmse(const float[:,::1] A, const float[:,::1] B) noexcept nogil:
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
cpdef void galinskyScan(const float[:,::1] U, float[:,::1] Dsquared, const int t) \
		noexcept nogil:
	cdef:
		int M = U.shape[0]
		int K = U.shape[1]
		int j, k
	for j in prange(M, num_threads=t):
		for k in range(K):
			Dsquared[j,k] = (U[j,k]**2)*float(M)

### Accelerated EM
cpdef void alphaStep(float[:,::1] X0, const float[:,::1] X1, const float[:,::1] X2) \
		noexcept nogil:
	cdef:
		int I = X0.shape[0]
		int J = X0.shape[1]
		int i, j
		float c1, c2
	c1 = min(max(computeC(&X0[0,0], &X1[0,0], &X2[0,0], I, J), 1.0), 256.0)
	c2 = 1.0 - c1
	for i in range(I):
		for j in range(J):
			X0[i,j] = c2*X1[i,j] + c1*X2[i,j]

### Likelihood measures for debugging
cpdef void frobenius(const unsigned char[:,::1] G, const float[::1] f, \
		const float[:,::1] U, const float[:,::1] V, float[::1] sumV, const int t) \
		noexcept nogil:
	cdef:
		int M = U.shape[0]
		int K = U.shape[1]
		int N = V.shape[0]
		int B = G.shape[1]
		int b, i, j, k, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
		float e
	for j in prange(M, num_threads=t):
		i = 0
		sumV[j] = 0.0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					e = 0.0
					for k in range(K):
						e = e + U[j,k]*V[i,k]
					e = project(e, f[j])
					sumV[j] += (<float>g - e)**2
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break
