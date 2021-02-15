import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt

# Typedef
DTYPE = np.float32
DTYPE2 = np.int32
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t DTYPE2_t

##### EMU #####
# Estimate population allele frequencies
@boundscheck(False)
@wraparound(False)
cpdef estimateF(unsigned char[:,::1] D, float[::1] f, int Bi, int n, int m, \
				int t):
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, b, bytepart
	cdef int[::1] c = np.zeros(m, dtype=DTYPE2)
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
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						f[j] = f[j]/float(c[j])
						break

# Array filtering
@boundscheck(False)
@wraparound(False)
cpdef filterArrays(unsigned char[:,::1] D, float[::1] f, \
					unsigned char[::1] mask):
	cdef int m = D.shape[0]
	cdef int n = D.shape[1]
	cdef int c = 0
	cdef int i, j
	for j in range(m):
		if mask[j] == 1:
			for i in range(n):
				D[c,i] = D[j,i] # Data matrix
			f[c] = f[j] # Allele frequency
			c += 1

# Initial update of dosage matrix
@boundscheck(False)
@wraparound(False)
cpdef updateE_init(unsigned char[:,::1] D, float[::1] f, float[:,::1] E, \
					int Bi, int t):
	cdef int m = E.shape[0]
	cdef int n = E.shape[1]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, b, bytepart
	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						if code == 2:
							E[j,i] = 0.5 - f[j]
						else:
							E[j,i] = code - f[j]
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Update E directly from SVD
@boundscheck(False)
@wraparound(False)
cpdef updateE_SVD(unsigned char[:,::1] D, float[:,::1] E, float[::1] f, \
					float[:,:] U, float[:] s, float[:,:] W, int Bi, int t):
	cdef int m = E.shape[0]
	cdef int n = E.shape[1]
	cdef int K = s.shape[0]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code == 9:
						E[j,i] = 0.0
						for k in range(K):
							E[j,i] += U[j,k]*s[k]*W[k,i]
						E[j,i] = min(max(E[j,i], -f[j]), 1-f[j])
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Accelerated update of E directly from SVD
@boundscheck(False)
@wraparound(False)
cpdef updateE_SVD_accel(unsigned char[:,::1] D, float[:,::1] E, float[::1] f, \
						float[:,:] U, float[:,:] SW, int Bi, int t):
	cdef int m = E.shape[0]
	cdef int n = E.shape[1]
	cdef int K = SW.shape[0]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code == 9:
						E[j,i] = 0.0
						for k in range(K):
							E[j,i] += U[j,k]*SW[k,i]
						E[j,i] = min(max(E[j,i], -f[j]), 1-f[j])
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Standardize dosage matrix
@boundscheck(False)
@wraparound(False)
cpdef standardizeMatrix(float[:,::1] E, float[::1] f, int t):
	cdef int m = E.shape[0]
	cdef int n = E.shape[1]
	cdef int i, j
	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for i in range(n):
				E[j,i] /= sqrt(f[j]*(1-f[j]))

# Root-mean squared error
@boundscheck(False)
@wraparound(False)
cpdef rmse(float[:,:] A, float[:,:] B):
	cdef int n = A.shape[0]
	cdef int m = A.shape[1]
	cdef int i, j
	cdef float res = 0.0
	for i in range(n):
			for j in range(m):
				res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
	return res

# Selection scan
@boundscheck(False)
@wraparound(False)
cpdef galinskyScan(float[:,:] U, float[:,::1] Dsquared):
	cdef int m = U.shape[0]
	cdef int K = U.shape[1]
	cdef int j, k
	for j in range(m):
		for k in range(K):
			Dsquared[j,k] = (U[j,k]**2)*float(m)

### Accelerated EM
# Matrix subtraction
@boundscheck(False)
@wraparound(False)
cpdef matMinus(float[:,:] M1, float[:,:] M2, float[:,::1] diffM):
	cdef int n = M1.shape[0]
	cdef int m = M1.shape[1]
	cdef int i, j
	for i in range(n):
		for j in range(m):
			diffM[i,j] = M1[i,j]-M2[i,j]

# Matrix sum of squares
@boundscheck(False)
@wraparound(False)
cpdef matSumSquare(float[:,:] M):
	cdef int n = M.shape[0]
	cdef int m = M.shape[1]
	cdef int i, j
	cdef float res = 0.0
	for i in range(n):
		for j in range(m):
			res += M[i,j]*M[i,j]
	return res

# Update factor matrices
@boundscheck(False)
@wraparound(False)
cpdef matUpdate(float[:,:] M, float[:,::1] diffM_1, float[:,::1] diffM_3, \
				float alpha):
	cdef int n = M.shape[0]
	cdef int m = M.shape[1]
	cdef int i, j
	for i in range(n):
		for j in range(m):
			M[i,j] = M[i,j] + 2*alpha*diffM_1[i,j] + alpha*alpha*diffM_3[i,j]

### Likelihood measures for debugging
@boundscheck(False)
@wraparound(False)
cpdef frobenius(unsigned char[:,::1] D, float[::1] f, float[:,:] U, float[:] s,\
				float[:,:] W, float[::1] sumVec, int Bi, int t):
	cdef int m = U.shape[0]
	cdef int n = W.shape[1]
	cdef int K = s.shape[0]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e
	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			sumVec[j] = 0.0
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						e = 0.0
						for k in range(K):
							e = e + U[j,k]*s[k]*W[k,i]
						e = e + f[j]
						e = min(max(e, 0), 1)
						if code == 2:
							sumVec[j] = sumVec[j] + (0.5 - e)**2
						else:
							sumVec[j] = sumVec[j] + (code - e)**2
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break

# Accelerated
@boundscheck(False)
@wraparound(False)
cpdef frobenius_accel(unsigned char[:,::1] D, float[::1] f, float[:,:] U, \
						float[:,:] SW, float[::1] sumVec, int Bi, int t):
	cdef int m = U.shape[0]
	cdef int n = SW.shape[1]
	cdef int K = SW.shape[0]
	cdef signed char[4] recode = [1, 9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef int i, j, k, b, bytepart
	cdef float e
	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			sumVec[j] = 0.0
			i = 0
			for b in range(Bi):
				byte = D[j,b]
				for bytepart in range(4):
					code = recode[byte & mask]
					if code != 9:
						e = 0.0
						for k in range(K):
							e = e + U[j,k]*SW[k,i]
						e = e + f[j]
						e = min(max(e, 0), 1)
						if code == 2:
							sumVec[j] = sumVec[j] + (0.5 - e)**2
						else:
							sumVec[j] = sumVec[j] + (code - e)**2
					byte = byte >> 2 # Right shift 2 bits
					i = i + 1
					if i == n:
						break
