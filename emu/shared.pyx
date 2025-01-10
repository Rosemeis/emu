# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### EMU #####
# Inline functions
cdef inline float project(const float e, const float f) noexcept nogil:
	return min(max(e + 2.0*f, 1e-4), 2.0-(1e-4))

cdef inline float innerE(const float* u, const float* v, const float f, \
		const size_t K) noexcept nogil:
	cdef:
		size_t k
		float e = 0.0
	for k in range(K):
		e += u[k]*v[k]
	return project(e, f) - 2.0*f

cdef inline float computeC(const float* x0, const float* x1, const float* x2, \
		const size_t I) noexcept nogil:
	cdef:
		size_t i
		float sum1 = 0.0
		float sum2 = 0.0
		float u, v
	for i in prange(I):
		u = x1[i] - x0[i]
		v = x2[i] - x1[i] - u
		sum1 += u*u
		sum2 += u*v
	return min(max(-(sum1/sum2), 1.0), 256.0)

cdef inline void updateAlpha(float* x0, const float* x1, const float* x2, \
		const float c1, const size_t I) noexcept nogil:
	cdef:
		size_t i
		float c2 = 1.0 - c1
	for i in prange(I):
		x0[i] = c2*x1[i] + c1*x2[i]

# Estimate population allele frequencies
cpdef void estimateF(const unsigned char[:,::1] G, float[::1] f, float[::1] d, \
		unsigned int[::1] n, const size_t N) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t B = G.shape[1]
		size_t b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
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
			if (f[j] > 0.0) and (f[j] < 1.0):
				d[j] = 1.0/sqrt(2.0*f[j]*(1.0 - f[j]))
		else:
			f[j] = 0.0

# Initialize and standardize E
cpdef void standardInit(const unsigned char[:,::1] G, float[:,::1] E, \
		float[::1] f, float[::1] d) noexcept nogil:
	cdef:
		size_t M = E.shape[0]
		size_t N = E.shape[1]
		size_t B = G.shape[1]
		size_t b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					E[j,i] = (<float>g - 2.0*f[j])*d[j]
				else:
					E[j,i] = 0.0
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Initialize and center E
cpdef void centerInit(const unsigned char[:,::1] G, float[:,::1] E, \
		float[::1] f) noexcept nogil:
	cdef:
		size_t M = E.shape[0]
		size_t N = E.shape[1]
		size_t B = G.shape[1]
		size_t b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					E[j,i] = <float>g - 2.0*f[j]
				else:
					E[j,i] = 0.0
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Standardize E in acceleration scheme
cpdef void standardAccel(const unsigned char[:,::1] G, float[:,::1] E, \
		float[:,::1] U, float[:,::1] V, float[::1] f, float[::1] d) noexcept nogil:
	cdef:
		size_t M = E.shape[0]
		size_t N = E.shape[1]
		size_t K = U.shape[1]
		size_t B = G.shape[1]
		size_t b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					E[j,i] = <float>g - 2.0*f[j]
				else:
					E[j,i] = innerE(&U[j,0], &V[i,0], f[j], K)
				E[j,i] *= d[j]
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Center E in acceleration scheme
cpdef void centerAccel(const unsigned char[:,::1] G, float[:,::1] E, \
		float[:,::1] U, float[:,::1] V, float[::1] f) noexcept nogil:
	cdef:
		size_t M = E.shape[0]
		size_t N = E.shape[1]
		size_t K = U.shape[1]
		size_t B = G.shape[1]
		size_t b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					E[j,i] = <float>g - 2.0*f[j]
				else:
					E[j,i] = innerE(&U[j,0], &V[i,0], f[j], K)
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Accelerated QN jump
cpdef void alphaStep(float[:,::1] X0, const float[:,::1] X1, const float[:,::1] X2) \
		noexcept nogil:
	cdef:
		size_t I = X0.shape[0]
		size_t J = X0.shape[1]
		float c
	c = computeC(&X0[0,0], &X1[0,0], &X2[0,0], I*J)
	updateAlpha(&X0[0,0], &X1[0,0], &X2[0,0], c, I*J)

# Root-mean squared error
cpdef float rmse(const float[:,::1] A, const float[:,::1] B) noexcept nogil:
	cdef:
		size_t M = A.shape[0]
		size_t K = A.shape[1]
		size_t j, k
		float res = 0.0
	for j in prange(M):
		for k in range(K):
			res += (A[j,k] - B[j,k])*(A[j,k] - B[j,k])
	return sqrt(res/(<float>(M*K)))

# Selection scan
cpdef void galinskyScan(const float[:,::1] U, float[:,::1] Dsquared) \
		noexcept nogil:
	cdef:
		int M = U.shape[0]
		int K = U.shape[1]
		int j, k
	for j in prange(M):
		for k in range(K):
			Dsquared[j,k] = (U[j,k]*U[j,k])*<float>M
