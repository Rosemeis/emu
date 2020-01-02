import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt

# Typedef
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

##### Functions for 2-bits #####
# Estimate population allele frequencies
@boundscheck(False)
@wraparound(False)
cpdef estimateF(unsigned char[:,::1] D, float[::1] f, int Bi, int n, int m, int t):
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, b, bytepart
	cdef int[:] c = np.zeros(m, dtype=DTYPE)

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						c[j] = c[j] + 1
						if code == 2:
							f[j] = f[j] + 0.5
						else:
							f[j] = f[j] + code

					byte = byte >> 2
					i = i + 1
					if i == n:
						f[j] = f[j]/float(c[j])
						break

# Estimate guided allele frequencies
@boundscheck(False)
@wraparound(False)
cpdef estimateF_guided(unsigned char[:,::1] D, float[::1] f, float[:,::1] F, signed char[::1] p, \
						int Bi, int n, int m, int t):
	cdef int K = F.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef int[:,:] C = np.zeros((m, K), dtype=DTYPE)

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						for k in range(K):
							if p[i] == k:
								C[j,k] = C[j,k] + 1
								if code == 2:
									F[j,k] = F[j,k] + 0.5
								else:
									F[j,k] = F[j,k] + code
								break

					byte = byte >> 2
					i = i + 1
					if i == n:
						for k in range(K):
							if C[j,k] < 5:
								F[j,k] = f[j]
							else:
								F[j,k] = F[j,k]/float(C[j,k])
						break


### Frequency functions
# Matrix Multiplication from byte matrix - dot(E, X)
@boundscheck(False)
@wraparound(False)
cpdef matMul_Freq(unsigned char[:,::1] D, float[::1] f, float[:,::1] X, float[:,::1] Y, \
					int Bi, int n, int m, int t):
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
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
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_Freq(unsigned char[:,::1] D, float[::1] f, float[:,:] Y, float[:,::1] X, \
						int Bi, int n, int m, int t):
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e

	with nogil:
		for b in prange(Bi, num_threads=t, schedule='static'):
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
@boundscheck(False)
@wraparound(False)
cpdef matMulFinal_Freq(unsigned char[:,::1] D, float[::1] f, float[:,:] X, float[:,:] Y, \
						int Bi, int n, int m, int t):
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
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
@boundscheck(False)
@wraparound(False)
cpdef matMulTransFinal_Freq(unsigned char[:,::1] D, float[::1] f, float[:,:] Y, float[:,::1] X, \
								int Bi, int n, int m, int t):
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e

	with nogil:
		for b in prange(Bi, num_threads=t, schedule='static'):
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
@boundscheck(False)
@wraparound(False)
cpdef matMul_SVD_domain(unsigned char[:,::1] D, float[::1] f, float[:,:] U, float[:] s, float[:,:] W, \
							float[:,::1] X, float[:,::1] Y, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
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
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_SVD_domain(unsigned char[:,::1] D, float[::1] f, float[:,:] U, float[:] s, float[:,:] W, \
								float[:,:] Y, float[:,::1] X, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e

	with nogil:
		for b in prange(Bi, num_threads=t, schedule='static'):
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
@boundscheck(False)
@wraparound(False)
cpdef matMulFinal_SVD(unsigned char[:,::1] D, float[::1] f, float[:,:] U, float[:] s, float[:,:] W, \
						float[:,:] X, float[:,:] Y, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
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
@boundscheck(False)
@wraparound(False)
cpdef matMulTransFinal_SVD(unsigned char[:,::1] D, float[::1] f, float[:,:] U, float[:] s, float[:,:] W, \
							float[:,:] Y, float[:,::1] X, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e

	with nogil:
		for b in prange(Bi, num_threads=t, schedule='static'):
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
@boundscheck(False)
@wraparound(False)
cpdef matMul_SVD_domain_accel(unsigned char[:,::1] D, float[::1] f, float[:,:] U, float[:,:] W, \
							float[:,::1] X, float[:,::1] Y, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
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
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_SVD_domain_accel(unsigned char[:,::1] D, float[::1] f, float[:,:] U, float[:,:] W, \
								float[:,:] Y, float[:,::1] X, int Bi, int n, int m, int t):
	cdef int V = U.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, v, b, bytepart
	cdef float e

	with nogil:
		for b in prange(Bi, num_threads=t, schedule='static'):
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


### Guided functions
# Matrix Multiplication from byte matrix - dot(E, X)
@boundscheck(False)
@wraparound(False)
cpdef matMul_Guide(unsigned char[:,::1] D, float[::1] f, float[:,::1] F, signed char[::1] p, \
					float[:,::1] X, float[:,::1] Y, int Bi, int n, int m, int t):
	cdef int V = F.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, v, k, b, bytepart
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
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
						if p[i] == -9:
							e = 0.0
						else:
							for v in range(V):
								if p[i] == v:
									e = F[j,v] - f[j]
									break
					for k in range(K):
						Y[j,k] = Y[j,k] + e*X[i,k]

					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Trans Matrix Multiplication from byte matrix - dot(E.T, Y)
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_Guide(unsigned char[:,::1] D, float[::1] f, float[:,::1] F, signed char[::1] p, \
							float[:,:] Y, float[:,::1] X, int Bi, int n, int m, int t):
	cdef int V = F.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, v, k, b, bytepart
	cdef float e

	with nogil:
		for b in prange(Bi, num_threads=t, schedule='static'):
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
						if p[i] == -9:
							e = 0.0
						else:
							for v in range(V):
								if p[i] == v:
									e = F[j,v] - f[j]
									break
					for k in range(K):
						X[i,k] = X[i,k] + e*Y[j,k]

					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

## Final iteration
# Matrix Multiplication from byte matrix - dot(E, X)
@boundscheck(False)
@wraparound(False)
cpdef matMulFinal_Guide(unsigned char[:,::1] D, float[::1] f, float[:,::1] F, signed char[::1] p, \
							float[:,:] X, float[:,:] Y, int Bi, int n, int m, int t):
	cdef int V = F.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, v, k, b, bytepart
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
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
						if p[i] == -9:
							e = 0.0
						else:
							for v in range(V):
								if p[i] == v:
									e = (F[j,v] - f[j])/sqrt(f[j]*(1 - f[j]))
									break
					for k in range(K):
						Y[j,k] = Y[j,k] + e*X[i,k]

					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Trans Matrix Multiplication from byte matrix - dot(E.T, Y)
@boundscheck(False)
@wraparound(False)
cpdef matMulTransFinal_Guide(unsigned char[:,::1] D, float[::1] f, float[:,::1] F, signed char[::1] p, \
								float[:,:] Y, float[:,::1] X, int Bi, int n, int m, int t):
	cdef int V = F.shape[1]
	cdef int K = X.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, v, k, b, bytepart
	cdef float e

	with nogil:
		for b in prange(Bi, num_threads=t, schedule='static'):
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
						if p[i] == -9:
							e = 0.0
						else:
							for v in range(V):
								if p[i] == v:
									e = (F[j,v] - f[j])/sqrt(f[j]*(1 - f[j]))
									break
					for k in range(K):
						X[i,k] = X[i,k] + e*Y[j,k]

					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break


### Galinsky selection scan
@boundscheck(False)
@wraparound(False)
cpdef galinskyScan(float[:,:] U, float[:,::1] Dsquared, int m, int K, int t):
	cdef int j, k
	
	# Loop over different PCs
	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for k in range(K):
				Dsquared[j,k] = (U[j,k]**2)*float(m)
	return Dsquared