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

# Estimate allele frequencies
@boundscheck(False)
@wraparound(False)
cpdef estimateF(signed char[:,::1] D, float[::1] f, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int i, j
	cdef int[:] c = np.zeros(m, dtype=DTYPE2)

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for i in range(n):
				if D[i,j] != -9:
					c[j] += 1
					f[j] += D[i,j]

			if c[j] == 0:
				f[j] = 0.0
			else:
				f[j] /= float(c[j])

	return f

# Estimate guided allele frequencies
@boundscheck(False)
@wraparound(False)
cpdef estimateF_guided(signed char[:,::1] D, float[::1] f, float[:,::1] F, signed char[::1] p, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int i, j, k
	cdef int K = max(p) + 1
	cdef int[:,:] C = np.zeros((m, K), dtype=DTYPE2)

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for i in range(n):
				for k in range(K):
					if p[i] == k:
						if D[i,j] != -9:
							C[j,k] += 1
							F[j,k] += D[i,j]
			
			for k in range(K):
				if C[j,k] < 5:
					F[j,k] = f[j]
				else:
					F[j,k] /= float(C[j,k])

# Initial update of dosage matrix
@boundscheck(False)
@wraparound(False)
cpdef updateE_init(signed char[:,::1] D, float[::1] f, float[:,::1] E, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int i, j

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for j in range(m):
				if D[i,j] == -9:
					E[i,j] = f[j]
				else:
					E[i,j] = D[i,j]

@boundscheck(False)
@wraparound(False)
cpdef updateE_init_guided(signed char[:,::1] D, float[:,::1] F, signed char[::1] p, float[:,::1] E, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int i, j, k
	cdef int K = F.shape[1]

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for j in range(m):
				for k in range(K):
					if p[i] == k: 
						if D[i, j] == -9:
							E[i,j] = F[j,k]
						else:
							E[i,j] = D[i,j]

# Update E directly from SVD
@boundscheck(False)
@wraparound(False)
cpdef updateE_SVD(signed char[:,::1] D, float[:,::1] E, float[::1] f, float[:,:] W, float[:] s, float[:,:] U, int t):
	cdef int n = E.shape[0]
	cdef int m = E.shape[1]
	cdef int K = s.shape[0]
	cdef int i, j, k

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for j in range(m):
				if D[i,j] == -9: # Missing site
					E[i,j] = 0.0
					for k in range(K):
						E[i,j] += W[i,k]*s[k]*U[k,j]
					E[i,j] += f[j]
					E[i,j] = min(max(E[i,j], 1e-4), 1-(1e-4))
				else:
					E[i,j] = D[i,j]

# Center dosage matrix
@boundscheck(False)
@wraparound(False)
cpdef centerMatrix(float[:,::1] E, float[::1] f, int t):
	cdef int n = E.shape[0]
	cdef int m = E.shape[1]
	cdef int i, j

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for j in range(m):
				E[i,j] -= f[j]

# Standardize dosage matrix
@boundscheck(False)
@wraparound(False)
cpdef standardizeMatrix(float[:,::1] E, float[::1] f, int t):
	cdef int n = E.shape[0]
	cdef int m = E.shape[1]
	cdef int i, j

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for j in range(m):
				E[i,j] -= f[j]
				E[i,j] /= sqrt(f[j]*(1-f[j]))

# RMSE inner C
cdef float rmseInner(float A, float B) nogil:
	cdef float val = 0.0
	if (A > 0) & (B > 0):
		val += (A - B)*(A - B)
	elif (A < 0) & (B < 0):
		val += (A - B)*(A - B)
	else:
		val += (A + B)*(A + B)

	return val

# Root-mean squared error
@boundscheck(False)
@wraparound(False)
cpdef rmse(float[:,:] A, float[:,:] B, int t):
	cdef int n = A.shape[0]
	cdef int m = A.shape[1]
	cdef int i, j
	cdef float[:] R = np.zeros(n, dtype=DTYPE)

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for j in range(m):
				R[i] += rmseInner(A[i,j], B[i,j])
	return R

# Selection scan
@boundscheck(False)
@wraparound(False)
cpdef galinskyScan(float[:,:] U, float[:,:] Dsquared, int t):
	cdef int e = U.shape[0]
	cdef int m = U.shape[1]
	cdef int j, k
	
	# Loop over different PCs
	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for k in range(e):
				Dsquared[j,k] = (U[k,j]**2)*float(m)

	return Dsquared