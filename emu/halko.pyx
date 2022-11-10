# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### EMU-mem #####
### 2-bit functions only ###
### Frequency functions
# Matrix Multiplication from byte matrix - dot(E, X)
cpdef matMul_Freq(unsigned char[:,::1] D, float[::1] f, float[:,::1] X, \
					float[:,::1] Y, int Bi, int n, int m, int t):
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e
	with nogil:
		for j in prange(m, num_threads=t):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = 0.5 - f[j]
						else:
							e = code - f[j]
						for k in range(K):
							Y[j,k] = Y[j,k] + e*X[i,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Trans Matrix Multiplication from byte matrix - dot(E.T, Y)
cpdef matMulTrans_Freq(unsigned char[:,::1] D, float[::1] f, float[:,:] Y, \
						float[:,::1] X, int Bi, int n, int m, int t):
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e
	with nogil:
		for b in prange(Bi, num_threads=t):
			for j in range(m):
				byte = D[j,b]
				i = b*4
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = 0.5 - f[j]
						else:
							e = code - f[j]
						for k in range(K):
							X[i,k] = X[i,k] + e*Y[j,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

## Final iteration
# Matrix Multiplication from byte matrix - dot(E, X)
cpdef matMulFinal_Freq(unsigned char[:,::1] D, float[::1] f, float[:,:] X, \
						float[:,:] Y, int Bi, int n, int m, int t):
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e
	with nogil:
		for j in prange(m, num_threads=t):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = (0.5 - f[j])/sqrt(f[j]*(1 - f[j]))
						else:
							e = (code - f[j])/sqrt(f[j]*(1 - f[j]))
						for k in range(K):
							Y[j,k] = Y[j,k] + e*X[i,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Trans Matrix Multiplication from byte matrix - dot(E.T, Y)
cpdef matMulTransFinal_Freq(unsigned char[:,::1] D, float[::1] f, float[:,:] Y,\
							float[:,::1] X, int Bi, int n, int m, int t):
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e
	with nogil:
		for b in prange(Bi, num_threads=t):
			for j in range(m):
				byte = D[j,b]
				i = b*4
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = (0.5 - f[j])/sqrt(f[j]*(1 - f[j]))
						else:
							e = (code - f[j])/sqrt(f[j]*(1 - f[j]))
						for k in range(K):
							X[i,k] = X[i,k] + e*Y[j,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

### SVD update functions
## Domain mapping
# Matrix Multiplication from byte matrix - dot(E, X)
cpdef matMul_SVD_domain(unsigned char[:,::1] D, float[::1] f, float[:,:] U, \
						float[:] s, float[:,:] W, float[:,::1] X, \
						float[:,::1] Y, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e
	with nogil:
		for j in prange(m, num_threads=t):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = 0.5 - f[j]
						else:
							e = code - f[j]
					else:
						e = 0.0
						for v in range(V):
							e = e + U[j,v]*s[v]*W[v,i]
						e = min(max(e + f[j], 1e-4), 1-(1e-4))
						e = e - f[j]
					for k in range(K):
						Y[j,k] = Y[j,k] + e*X[i,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Trans Matrix Multiplication from byte matrix - dot(E.T, Y)
cpdef matMulTrans_SVD_domain(unsigned char[:,::1] D, float[::1] f, \
								float[:,:] U, float[:] s, float[:,:] W, \
								float[:,:] Y, float[:,::1] X, int Bi, int n, \
								int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e
	with nogil:
		for b in prange(Bi, num_threads=t):
			for j in range(m):
				byte = D[j,b]
				i = b*4
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = 0.5 - f[j]
						else:
							e = code - f[j]
					else:
						e = 0.0
						for v in range(V):
							e = e + U[j,v]*s[v]*W[v,i]
						e = min(max(e + f[j], 1e-4), 1-(1e-4))
						e = e - f[j]
					for k in range(K):
						X[i,k] = X[i,k] + e*Y[j,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

## Final iteration
# Matrix Multiplication from byte matrix - dot(E, X)
cpdef matMulFinal_SVD(unsigned char[:,::1] D, float[::1] f, float[:,:] U, \
						float[:] s, float[:,:] W, float[:,:] X, float[:,:] Y, \
						int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e
	with nogil:
		for j in prange(m, num_threads=t):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = (0.5 - f[j])/sqrt(f[j]*(1 - f[j]))
						else:
							e = (code - f[j])/sqrt(f[j]*(1 - f[j]))
					else:
						e = 0.0
						for v in range(V):
							e = e + U[j,v]*s[v]*W[v,i]
						e = e/sqrt(f[j]*(1 - f[j]))
					for k in range(K):
						Y[j,k] = Y[j,k] + e*X[i,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Trans Matrix Multiplication from byte matrix - dot(E.T, Y)
cpdef matMulTransFinal_SVD(unsigned char[:,::1] D, float[::1] f, float[:,:] U, \
							float[:] s, float[:,:] W, float[:,:] Y, \
							float[:,::1] X, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e
	with nogil:
		for b in prange(Bi, num_threads=t):
			for j in range(m):
				byte = D[j,b]
				i = b*4
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = (0.5 - f[j])/sqrt(f[j]*(1 - f[j]))
						else:
							e = (code - f[j])/sqrt(f[j]*(1 - f[j]))
					else:
						e = 0.0
						for v in range(V):
							e = e + U[j,v]*s[v]*W[v,i]
						e = e/sqrt(f[j]*(1 - f[j]))
					for k in range(K):
						X[i,k] = X[i,k] + e*Y[j,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

### Acceleration functions
## Map2Domain
# Matrix Multiplication from byte matrix - dot(E, X)
cpdef matMul_SVD_domain_accel(unsigned char[:,::1] D, float[::1] f, \
								float[:,:] U, float[:,:] W, float[:,::1] X, \
								float[:,::1] Y, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e
	with nogil:
		for j in prange(m, num_threads=t):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = 0.5 - f[j]
						else:
							e = code - f[j]
					else:
						e = 0.0
						for v in range(V):
							e = e + U[j,v]*W[v,i]
						e = min(max(e + f[j], 1e-4), 1-(1e-4))
						e = e - f[j]
					for k in range(K):
						Y[j,k] = Y[j,k] + e*X[i,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Trans Matrix Multiplication from byte matrix - dot(E.T, Y)
cpdef matMulTrans_SVD_domain_accel(unsigned char[:,::1] D, float[::1] f, \
									float[:,:] U, float[:,:] W, float[:,:] Y, \
									float[:,::1] X, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e
	with nogil:
		for b in prange(Bi, num_threads=t):
			for j in range(m):
				byte = D[j,b]
				i = b*4
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							e = 0.5 - f[j]
						else:
							e = code - f[j]
					else:
						e = 0.0
						for v in range(V):
							e = e + U[j,v]*W[v,i]
						e = min(max(e + f[j], 1e-4), 1-(1e-4))
						e = e - f[j]
					for k in range(K):
						X[i,k] = X[i,k] + e*Y[j,k]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break
