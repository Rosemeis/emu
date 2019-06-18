import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt, fabs

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
	cdef int K = F.shape[1]
	cdef int[:,:] C = np.zeros((m, K), dtype=DTYPE2)

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for i in range(n):
				if D[i,j] != -9:
					for k in range(K):
						if p[i] == k:
							C[j,k] += 1
							F[j,k] += D[i,j]
							break
			
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
					E[i,j] = 0
				else:
					E[i,j] = D[i,j] - f[j]

@boundscheck(False)
@wraparound(False)
cpdef updateE_init_guided(signed char[:,::1] D, float[::1] f, float[:,::1] F, signed char[::1] p, float[:,::1] E, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int i, j, k
	cdef int K = F.shape[1]

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for j in range(m):
				if D[i,j] == -9:
					if p[i] == -9:
						E[i,j] = 0
					else:
						for k in range(K):
							if p[i] == k: 
								E[i,j] = F[j,k] - f[j]
								break
				else:
					E[i,j] = D[i,j] - f[j]

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
					E[i,j] -= f[j]
				else:
					E[i,j] = D[i,j] - f[j]

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
				E[i,j] /= sqrt(f[j]*(1-f[j]))

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
				R[i] += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
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

### Accelerated EM
# Matrix subtraction
@boundscheck(False)
@wraparound(False)
cpdef matMinus(float[:,:] M1, float[:,:] M2, float[:,:] diffM):
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

# Update E without D and f for two additional EM steps
@boundscheck(False)
@wraparound(False)
cpdef updateE_SVD_accel(signed char[:,::1] D, float[:,::1] E, float[::1] f, float[:,:] Ws, float[:,:] U, int t):
	cdef int n = E.shape[0]
	cdef int m = E.shape[1]
	cdef int K = Ws.shape[1]
	cdef int i, j, k

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for j in range(m):
				if D[i,j] == -9: # Missing site
					E[i,j] = 0.0
					for k in range(K):
						E[i,j] += Ws[i,k]*U[k,j]
					E[i,j] += f[j]
					E[i,j] = min(max(E[i,j], 1e-4), 1-(1e-4))
					E[i,j] = E[i,j] - f[j]
				else:
					E[i,j] = D[i,j] - f[j]

@boundscheck(False)
@wraparound(False)
cpdef updateE_SVD_accel2(signed char[:,::1] D, float[:,::1] E, float[::1] f, float[:,:] Ws, float[:,:] U, int t):
	cdef int n = E.shape[0]
	cdef int m = E.shape[1]
	cdef int K = Ws.shape[1]
	cdef int i, j, k

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for j in range(m):
				if D[i,j] == -9: # Missing site
					E[i,j] = 0.0
					for k in range(K):
						E[i,j] += Ws[i,k]*U[k,j]
				else:
					E[i,j] = D[i,j] - f[j]


@boundscheck(False)
@wraparound(False)
cpdef matUpdate(float[:,:] M, float[:,:] diffM_1, float[:,:] diffM_3, float alpha):
	cdef int n = M.shape[0]
	cdef int m = M.shape[1]
	cdef int i, j

	for i in range(n):
		for j in range(m):
			M[i,j] = M[i,j] + 2*alpha*diffM_1[i,j] + alpha*alpha*diffM_3[i,j]


# Likelihood measure
@boundscheck(False)
@wraparound(False)
cpdef frobenius(signed char[:,::1] D, float[::1] f, float[:,:] W, float[:] s, float[:,:] U, float[::1] sumVec, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int K = s.shape[0]
	cdef int i, j, k
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			sumVec[i] = 0.0
			for j in range(m):
				if D[i,j] != -9:
					e = 0.0
					for k in range(K):
						e = e + W[i,k]*s[k]*U[k,j]
					e = e + f[j]
					e = min(max(e, 0), 1)
					sumVec[i] = sumVec[i] + (D[i,j] - e)**2


@boundscheck(False)
@wraparound(False)
cpdef frobenius_accel(signed char[:,::1] D, float[::1] f, float[:,:] Ws, float[:,:] U, float[::1] sumVec, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int K = Ws.shape[1]
	cdef int i, j, k
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			sumVec[i] = 0.0
			for j in range(m):
				if D[i,j] != -9:
					e = 0.0
					for k in range(K):
						e = e + Ws[i,k]*U[k,j]
					e = e + f[j]
					e = min(max(e, 0), 1)
					sumVec[i] = sumVec[i] + (D[i,j] - e)**2