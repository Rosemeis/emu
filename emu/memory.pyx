# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### EMU-mem #####
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

# Extract and center chunk (frequencies) for randomized SVD
cpdef void memCenter(const unsigned char[:,::1] G, float[:,::1] X, float[::1] f, \
		const size_t M_w) noexcept nogil:
	cdef:
		size_t M = X.shape[0]
		size_t N = X.shape[1]
		size_t B = G.shape[1]
		size_t b, i, j, l, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
		l = M_w + j
		i = 0
		for b in range(B):
			byte = G[l,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					X[j,i] = g - 2.0*f[l]
				else:
					X[j,i] = 0.0
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Extract and center chunk (SVD) for randomized SVD
cpdef void memCenterSVD(const unsigned char[:,::1] G, const float[:,::1] U, \
		const float[:,::1] V, float[:,::1] X, float[::1] f, const size_t M_w) \
		noexcept nogil:
	cdef:
		size_t M = X.shape[0]
		size_t N = X.shape[1]
		size_t B = G.shape[1]
		size_t K = U.shape[1]
		size_t b, i, j, k, l, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
		l = M_w + j
		i = 0
		for b in range(B):
			byte = G[l,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					X[j,i] = g - 2.0*f[l]
				else:
					X[j,i] = innerE(&U[l,0], &V[i,0], f[l], K)
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Extract and standardize chunk (frequencies) for randomized SVD
cpdef void memFinal(const unsigned char[:,::1] G, float[:,::1] X, float[::1] f, \
		float[::1] d, const size_t M_w) noexcept nogil:
	cdef:
		size_t M = X.shape[0]
		size_t N = X.shape[1]
		size_t B = G.shape[1]
		size_t b, i, j, l, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
		l = M_w + j
		i = 0
		for b in range(B):
			byte = G[l,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					X[j,i] = (g - 2.0*f[l])*d[l]
				else:
					X[j,i] = 0.0
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Extract and standardize chunk (SVD) for randomized SVD
cpdef void memFinalSVD(const unsigned char[:,::1] G, const float[:,::1] U, \
		const float[:,::1] V, float[:,::1] X, float[::1] f, float[::1] d, \
		const size_t M_w) noexcept nogil:
	cdef:
		size_t M = X.shape[0]
		size_t N = X.shape[1]
		size_t B = G.shape[1]
		size_t K = U.shape[1]
		size_t b, i, j, k, l, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char g, byte
	for j in prange(M):
		l = M_w + j
		i = 0
		for b in range(B):
			byte = G[l,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					X[j,i] = g - 2.0*f[l]
				else:
					X[j,i] = innerE(&U[l,0], &V[i,0], f[l], K)
				X[j,i] *= d[l]
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break
