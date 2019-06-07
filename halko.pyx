import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt

### Custom Halko functions (based on the scikit-learn implementation)
## Centered computations
# dot(E, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMul_SVD(signed char[:,::1] D, float[::1] f, float[:,:] W, float[:] s, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int K = s.shape[0]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for o in range(O):
				C[i,o]
				for j in range(m):
					if D[i,j] == -9:
						e = 0.0
						for k in range(K):
							e = e + W[i,k]*s[k]*U[k,j]
					else:
						e = D[i,j] - f[j]
					C[i,o] = C[i,o] + e*B[j,o]

# dot(E.T, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_SVD(signed char[:,::1] Dt, float[::1] f, float[:,:] W, float[:] s, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int m = Dt.shape[0]
	cdef int n = Dt.shape[1]
	cdef int K = s.shape[0]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for o in range(O):
				C[j,o] = 0.0
				for i in range(n):
					if Dt[j,i] == -9:
						e = 0.0
						for k in range(K):
							e = e + U[j,k]*s[k]*W[k,i]
					else:
						e = Dt[j,i] - f[j]
					C[j,o] = C[j,o] + e*B[i,o]

# dot(E, Q) - Freq
@boundscheck(False)
@wraparound(False)
cpdef matMul_Freq(signed char[:,::1] D, float[::1] f, float[:,:] B, float[:,:] C, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for o in range(O):
				C[i,o] = 0.0
				for j in range(m):
					if D[i,j] == -9:
						e = 0.0
					else:
						e = D[i,j] - f[j]
					C[i,o] = C[i,o] + e*B[j,o]

# dot(E.T, Q) - Freq
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_Freq(signed char[:,::1] Dt, float[::1] f, float[:,:] B, float[:,:] C, int t):
	cdef int m = Dt.shape[0]
	cdef int n = Dt.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for o in range(O):
				C[j,o] = 0.0
				for i in range(n):
					if Dt[j,i] == -9:
						e = 0.0
					else:
						e = Dt[j,i] - f[j]
					C[j,o] = C[j,o] + e*B[i,o]

# dot(E, Q) - Guide
@boundscheck(False)
@wraparound(False)
cpdef matMul_Guide(signed char[:,::1] D, float[::1] f, float[:,::1] F, signed char[:] p, float[:,:] B, float[:,:] C, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int K = F.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for o in range(O):
				C[i,o] = 0.0
				for j in range(m):
					if D[i,j] == -9:
						for k in range(K):
							if p[i] == k:
								e = F[j,k] - f[j]
								break
					else:
						e = D[i,j] - f[j]
					C[i,o] = C[i,o] + e*B[j,o]

# dot(E.T, Q) - Guide
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_Guide(signed char[:,::1] Dt, float[::1] f, float[:,::1] F, signed char[:] p, float[:,:] B, float[:,:] C, int t):
	cdef int m = Dt.shape[0]
	cdef int n = Dt.shape[1]
	cdef int K = F.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for o in range(O):
				C[j,o] = 0.0
				for i in range(n):
					if Dt[j,i] == -9:
						for k in range(K):
							if p[i] == k:
								e = F[j,k] - f[j]
								break
					else:
						e = Dt[j,i] - f[j]
					C[j,o] = C[j,o] + e*B[i,o]


## Mapped back to domain
# dot(E, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMul_SVD_domain(signed char[:,::1] D, float[::1] f, float[:,:] W, float[:] s, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int K = s.shape[0]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for o in range(O):
				C[i,o]
				for j in range(m):
					if D[i,j] == -9:
						e = 0.0
						for k in range(K):
							e = e + W[i,k]*s[k]*U[k,j]
						e = e + f[j]
						e = min(max(e, 1e-4), 1-(1e-4))
						e = e - f[j]
					else:
						e = D[i,j] - f[j]
					C[i,o] = C[i,o] + e*B[j,o]

# dot(E.T, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_SVD_domain(signed char[:,::1] Dt, float[::1] f, float[:,:] W, float[:] s, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int m = Dt.shape[0]
	cdef int n = Dt.shape[1]
	cdef int K = s.shape[0]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for o in range(O):
				C[j,o] = 0.0
				for i in range(n):
					if Dt[j,i] == -9:
						e = 0.0
						for k in range(K):
							e = e + U[j,k]*s[k]*W[k,i]
						e = e + f[j]
						e = min(max(e, 1e-4), 1-(1e-4))
						e = e - f[j]
					else:
						e = Dt[j,i] - f[j]
					C[j,o] = C[j,o] + e*B[i,o]


## Standardized computations for final output
# dot(E, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMulFinal_SVD(signed char[:,::1] D, float[::1] f, float[:,:] W, float[:] s, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int K = s.shape[0]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for o in range(O):
				C[i,o]
				for j in range(m):
					if D[i,j] == -9:
						e = 0.0
						for k in range(K):
							e = e + W[i,k]*s[k]*U[k,j]
						e = e/(f[j]*(1 - f[j]))
					else:
						e = (D[i,j] - f[j])/(f[j]*(1 - f[j]))
					C[i,o] = C[i,o] + e*B[j,o]

# dot(E.T, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMulTransFinal_SVD(signed char[:,::1] Dt, float[::1] f, float[:,:] W, float[:] s, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int m = Dt.shape[0]
	cdef int n = Dt.shape[1]
	cdef int K = s.shape[0]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for o in range(O):
				C[j,o] = 0.0
				for i in range(n):
					if Dt[j,i] == -9:
						e = 0.0
						for k in range(K):
							e = e + U[j,k]*s[k]*W[k,i]
						e = e/(f[j]*(1 - f[j]))
					else:
						e = (Dt[j,i] - f[j])/(f[j]*(1 - f[j]))
					C[j,o] = C[j,o] + e*B[i,o]

# dot(E, Q) - Freq
@boundscheck(False)
@wraparound(False)
cpdef matMulFinal_Freq(signed char[:,::1] D, float[::1] f, float[:,:] B, float[:,:] C, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for o in range(O):
				C[i,o] = 0.0
				for j in range(m):
					if D[i,j] == -9:
						e = 0.0
					else:
						e = (D[i,j] - f[j])/(f[j]*(1 - f[j]))
					C[i,o] = C[i,o] + e*B[j,o]

# dot(E.T, Q) - Freq
@boundscheck(False)
@wraparound(False)
cpdef matMulTransFinal_Freq(signed char[:,::1] Dt, float[::1] f, float[:,:] B, float[:,:] C, int t):
	cdef int m = Dt.shape[0]
	cdef int n = Dt.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for o in range(O):
				C[j,o] = 0.0
				for i in range(n):
					if Dt[j,i] == -9:
						e = 0.0
					else:
						e = (Dt[j,i] - f[j])/(f[j]*(1 - f[j]))
					C[j,o] = C[j,o] + e*B[i,o]

# dot(E, Q) - Guide
@boundscheck(False)
@wraparound(False)
cpdef matMulFinal_Guide(signed char[:,::1] D, float[::1] f, float[:,::1] F, signed char[:] p, float[:,:] B, float[:,:] C, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int K = F.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for o in range(O):
				C[i,o] = 0.0
				for j in range(m):
					if D[i,j] == -9:
						for k in range(K):
							if p[i] == k:
								e = F[j,k] - f[j]
								break
						e = e/(f[j]*(1 - f[j]))
					else:
						e = (D[i,j] - f[j])/(f[j]*(1 - f[j]))
					C[i,o] = C[i,o] + e*B[j,o]

# dot(E.T, Q) - Guide
@boundscheck(False)
@wraparound(False)
cpdef matMulTransFinal_Guide(signed char[:,::1] Dt, float[::1] f, float[:,::1] F, signed char[:] p, float[:,:] B, float[:,:] C, int t):
	cdef int m = Dt.shape[0]
	cdef int n = Dt.shape[1]
	cdef int K = F.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for o in range(O):
				C[j,o] = 0.0
				for i in range(n):
					if Dt[j,i] == -9:
						for k in range(K):
							if p[i] == k:
								e = F[j,k] - f[j]
								break
						e = e/(f[j]*(1 - f[j]))
					else:
						e = (Dt[j,i] - f[j])/(f[j]*(1 - f[j]))
					C[j,o] = C[j,o] + e*B[i,o]


### Acceleration functions
# dot(E, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMul_SVD_accel(signed char[:,::1] D, float[::1] f, float[:,:] Ws, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int K = Ws.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for o in range(O):
				C[i,o]
				for j in range(m):
					if D[i,j] == -9:
						e = 0.0
						for k in range(K):
							e = e + Ws[i,k]*U[k,j]
					else:
						e = D[i,j] - f[j]
					C[i,o] = C[i,o] + e*B[j,o]

# dot(E.T, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_SVD_accel(signed char[:,::1] Dt, float[::1] f, float[:,:] Ws, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int m = Dt.shape[0]
	cdef int n = Dt.shape[1]
	cdef int K = U.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for o in range(O):
				C[j,o] = 0.0
				for i in range(n):
					if Dt[j,i] == -9:
						e = 0.0
						for k in range(K):
							e = e + U[j,k]*Ws[k,i]
					else:
						e = Dt[j,i] - f[j]
					C[j,o] = C[j,o] + e*B[i,o]


## Map2Domain
# dot(E, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMul_SVD_domain_accel(signed char[:,::1] D, float[::1] f, float[:,:] Ws, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int K = Ws.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for i in prange(n, num_threads=t, schedule='static'):
			for o in range(O):
				C[i,o]
				for j in range(m):
					if D[i,j] == -9:
						e = 0.0
						for k in range(K):
							e = e + Ws[i,k]*U[k,j]
						e = e + f[j]
						e = min(max(e, 1e-4), 1-(1e-4))
						e = e - f[j]
					else:
						e = D[i,j] - f[j]
					C[i,o] = C[i,o] + e*B[j,o]

# dot(E.T, Q) - SVD
@boundscheck(False)
@wraparound(False)
cpdef matMulTrans_SVD_domain_accel(signed char[:,::1] Dt, float[::1] f, float[:,:] Ws, float[:,:] U, float[:,:] B, float[:,:] C, int t):
	cdef int m = Dt.shape[0]
	cdef int n = Dt.shape[1]
	cdef int K = U.shape[1]
	cdef int O = C.shape[1]
	cdef int i, j, k, o
	cdef float e

	with nogil:
		for j in prange(m, num_threads=t, schedule='static'):
			for o in range(O):
				C[j,o] = 0.0
				for i in range(n):
					if Dt[j,i] == -9:
						e = 0.0
						for k in range(K):
							e = e + U[j,k]*Ws[k,i]
						e = e + f[j]
						e = min(max(e, 1e-4), 1-(1e-4))
						e = e - f[j]
					else:
						e = Dt[j,i] - f[j]
					C[j,o] = C[j,o] + e*B[i,o]