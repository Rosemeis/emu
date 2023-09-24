# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### EMU-mem #####
# Load centered chunk of PLINK file for SVD using frequencies
cpdef void plinkFreq(unsigned char[:,::1] D, float[:,::1] E, float[::1] f, \
		int M_w, int t) nogil:
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
			byte = D[M_w+j,b]
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
cpdef void plinkFinalFreq(unsigned char[:,::1] D, float[:,::1] E, float[::1] f, \
		int M_w, int t) nogil:
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
			byte = D[M_w+j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					E[j,i] = <float>recode[byte & mask] - 2.0*f[M_w+j]
					E[j,i] /= sqrt(2.0*f[M_w+j]*(1.0 - f[M_w+j]))
				else:
					E[j,i] = 0.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break

# Load centered chunk of PLINK file for SVD using factor matrices
cpdef void plinkSVD(unsigned char[:,::1] D, float[:,::1] E, float[:,::1] U, \
		float[:,::1] V, float[::1] f, int M_w, int t) nogil:
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
			byte = D[M_w+j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					E[j,i] = <float>recode[byte & mask] - 2.0*f[M_w+j]
				else:
					E[j,i] = 0.0
					for k in range(K):
						E[j,i] += U[M_w+j,k]*V[i,k]
					E[j,i] = min(max(E[j,i] + 2.0*f[M_w+j], 1e-4), 2-(1e-4))
					E[j,i] -= 2.0*f[M_w+j]
				byte = byte >> 2
				i = i + 1
				if i == N:
					break

# Load standardized chunk of PLINK file for SVD using factor matrices
cpdef void plinkFinalSVD(unsigned char[:,::1] D, float[:,::1] E, float[:,::1] U, \
		float[::1] S, float[:,::1] V, float[::1] f, int M_w, int t) nogil:
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
			byte = D[M_w+j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					E[j,i] = <float>recode[byte & mask] - 2.0*f[M_w+j]
				else:
					E[j,i] = 0.0
					for k in range(K):
						E[j,i] += U[M_w+j,k]*S[k]*V[i,k]
					E[j,i] = min(max(E[j,i] + 2.0*f[M_w+j], 1e-4), 2-(1e-4))
					E[j,i] -= 2.0*f[M_w+j]
				E[j,i] /= sqrt(2.0*f[M_w+j]*(1.0 - f[M_w+j]))
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
