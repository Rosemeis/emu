# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### EMU-mem #####
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

# Load centered chunk of PLINK file for SVD using frequencies
cpdef void plinkFreq(const unsigned char[:,::1] G, float[:,::1] E, \
		const float[::1] f, const int M_w, const int t) noexcept nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int B = G.shape[1]
		int b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = G[M_w+j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					E[j,i] = <float>recode[byte & mask] - 2.0*f[M_w+j]
				else:
					E[j,i] = 0.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break

# Load standardized chunk of PLINK file for SVD using frequencies
cpdef void plinkFinalFreq(const unsigned char[:,::1] G, float[:,::1] E, \
		const float[::1] f, const float[::1] d, const int M_w, const int t) \
		noexcept nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int B = G.shape[1]
		int b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = G[M_w+j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					E[j,i] = <float>recode[byte & mask] - 2.0*f[M_w+j]
					E[j,i] *= d[M_w+j]
				else:
					E[j,i] = 0.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break

# Load centered chunk of PLINK file for SVD using factor matrices
cpdef void plinkSVD(const unsigned char[:,::1] G, float[:,::1] E, \
		const float[:,::1] U, const float[:,::1] V, const float[::1] f, \
		const int M_w, const int t) noexcept nogil:
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
			byte = G[M_w+j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					E[j,i] = <float>recode[byte & mask] - 2.0*f[M_w+j]
				else:
					E[j,i] = innerAccelE(&U[M_w+j,0], &V[i,0], f[M_w+j], K)
				byte = byte >> 2
				i = i + 1
				if i == N:
					break

# Load standardized chunk of PLINK file for SVD using factor matrices
cpdef void plinkFinalSVD(const unsigned char[:,::1] G, float[:,::1] E, \
		const float[:,::1] U, const float[::1] S, const float[:,::1] V, \
		const float[::1] f, const float[::1] d, const int M_w, const int t) \
		noexcept nogil:
	cdef:
		int M = E.shape[0]
		int N = E.shape[1]
		int B = G.shape[1]
		int K = U.shape[1]
		int i, j, k, b, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(B):
			byte = G[M_w+j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					E[j,i] = <float>recode[byte & mask] - 2.0*f[M_w+j]
				else:
					E[j,i] = innerE(&U[M_w+j,0], &S[0], &V[i,0], f[M_w+j], K)
				E[j,i] *= d[M_w+j]
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
