# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrtf
from libc.stdint cimport uint8_t, uint32_t

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef float f32

cdef f32 PRO_MIN = 1e-4
cdef f32 PRO_MAX = 2.0 - (1e-4)
cdef f32 ACC_MIN = 1.0
cdef f32 ACC_MAX = 256.0
cdef inline f32 _fmax(f32 a, f32 b) noexcept nogil: return a if a > b else b
cdef inline f32 _fmin(f32 a, f32 b) noexcept nogil: return a if a < b else b
cdef inline f32 _clamp1(f32 a) noexcept nogil: return _fmax(PRO_MIN, _fmin(a, PRO_MAX))
cdef inline f32 _clamp2(f32 a) noexcept nogil: return _fmax(ACC_MIN, _fmin(a, ACC_MAX))


##### EMU-mem #####
# Inline functions
cdef inline f32 innerE(
		const f32* u, const f32* v, const f32 f, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		f32 d = 2.0*f
		f32 e = d
	for k in range(K):
		e += u[k]*v[k]
	return _clamp1(e) - d

cdef inline f32 computeC(
		const f32* x0, const f32* x1, const f32* x2, const Py_ssize_t I
	) noexcept nogil:
	cdef:
		size_t i
		f32 sum1 = 0.0
		f32 sum2 = 0.0
		f32 c, u, v
	for i in prange(I):
		u = x1[i] - x0[i]
		v = x2[i] - x1[i] - u
		sum1 += u*u
		sum2 += u*v
	c = -(sum1/sum2)
	return _clamp2(c)

cdef inline void updateAlpha(
		f32* x0, const f32* x1, const f32* x2, const f32 c1, const Py_ssize_t I
	) noexcept nogil:
	cdef:
		size_t i
		f32 c2 = 1.0 - c1
	for i in prange(I):
		x0[i] = c2*x1[i] + c1*x2[i]

# Estimate population allele frequencies
cpdef void estimateF(
		const u8[:,::1] G, f32[::1] f, f32[::1] d, u32[::1] n, const Py_ssize_t N
	) noexcept nogil:
	cdef:
		Py_ssize_t M = G.shape[0]
		Py_ssize_t B = G.shape[1]
		size_t b, i, j, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 g, byte
	for j in prange(M, schedule='guided'):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					f[j] += <f32>g
					n[j] += 1
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break
		if n[j] > 0:
			f[j] /= <f32>(2*n[j])
			if (f[j] > 0.0) and (f[j] < 1.0):
				d[j] = 1.0/sqrtf(2.0*f[j]*(1.0 - f[j]))
		else:
			f[j] = 0.0

# Initialize and standardize E
cpdef void standardInit(
		const u8[:,::1] G, f32[:,::1] E, f32[::1] f, f32[::1] d
	) noexcept nogil:
	cdef:
		Py_ssize_t M = E.shape[0]
		Py_ssize_t N = E.shape[1]
		Py_ssize_t B = G.shape[1]
		size_t b, i, j, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 g, byte
		f32 fj, dj
	for j in prange(M, schedule='guided'):
		i = 0
		fj = f[j]
		dj = d[j]
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					E[j,i] = (<f32>g - 2.0*fj)*dj
				else:
					E[j,i] = 0.0
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Initialize and center E
cpdef void centerInit(
		const u8[:,::1] G, f32[:,::1] E, f32[::1] f
	) noexcept nogil:
	cdef:
		Py_ssize_t M = E.shape[0]
		Py_ssize_t N = E.shape[1]
		Py_ssize_t B = G.shape[1]
		size_t b, i, j, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 g, byte
		f32 fj
	for j in prange(M, schedule='guided'):
		i = 0
		fj = f[j]
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					E[j,i] = <f32>g - 2.0*fj
				else:
					E[j,i] = 0.0
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Standardize E in acceleration scheme
cpdef void standardAccel(
		const u8[:,::1] G, f32[:,::1] E, f32[:,::1] U, f32[:,::1] V, f32[::1] f, f32[::1] d
	) noexcept nogil:
	cdef:
		Py_ssize_t M = E.shape[0]
		Py_ssize_t N = E.shape[1]
		Py_ssize_t K = U.shape[1]
		Py_ssize_t B = G.shape[1]
		size_t b, i, j, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 g, byte
		f32 fj, dj
		f32* Uj
	for j in prange(M, schedule='guided'):
		i = 0
		fj = f[j]
		dj = d[j]
		Uj = &U[j,0]
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					E[j,i] = (<f32>g - 2.0*fj)*dj
				else:
					E[j,i] = innerE(Uj, &V[i,0], fj, K)*dj
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Center E in acceleration scheme
cpdef void centerAccel(
		const u8[:,::1] G, f32[:,::1] E, f32[:,::1] U, f32[:,::1] V, f32[::1] f
	) noexcept nogil:
	cdef:
		Py_ssize_t M = E.shape[0]
		Py_ssize_t N = E.shape[1]
		Py_ssize_t K = U.shape[1]
		Py_ssize_t B = G.shape[1]
		size_t b, i, j, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 g, byte
		f32 fj
		f32* Uj
	for j in prange(M, schedule='guided'):
		i = 0
		fj = f[j]
		Uj = &U[j,0]
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					E[j,i] = <f32>g - 2.0*fj
				else:
					E[j,i] = innerE(Uj, &V[i,0], fj, K)
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Accelerated QN jump
cpdef void alphaStep(
		f32[:,::1] X0, const f32[:,::1] X1, const f32[:,::1] X2
	) noexcept nogil:
	cdef:
		Py_ssize_t I = X0.shape[0]
		Py_ssize_t J = X0.shape[1]
		f32 c
	c = computeC(&X0[0,0], &X1[0,0], &X2[0,0], I*J)
	updateAlpha(&X0[0,0], &X1[0,0], &X2[0,0], c, I*J)

# Root-mean squared error
cpdef f32 rmse(
		const f32[:,::1] A, const f32[:,::1] B
	) noexcept nogil:
	cdef:
		Py_ssize_t M = A.shape[0]
		Py_ssize_t K = A.shape[1]
		size_t j, k
		f32 res = 0.0
	for j in prange(M, schedule='guided'):
		for k in range(K):
			res += (A[j,k] - B[j,k])*(A[j,k] - B[j,k])
	return sqrtf(res/(<f32>(M*K)))

# Selection scan
cpdef void galinskyScan(
		const f32[:,::1] U, f32[:,::1] Dsquared
	) noexcept nogil:
	cdef:
		Py_ssize_t M = U.shape[0]
		Py_ssize_t K = U.shape[1]
		size_t j, k
		f32 m = <f32>M
	for j in prange(M, schedule='guided'):
		for k in range(K):
			Dsquared[j,k] = (U[j,k]*U[j,k])*m
