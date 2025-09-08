# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdint cimport uint8_t

ctypedef uint8_t u8
ctypedef float f32

cdef f32 PRO_MIN = 1e-4
cdef f32 PRO_MAX = 2.0 - (1e-4)
cdef inline f32 _fmax(f32 a, f32 b) noexcept nogil: return a if a > b else b
cdef inline f32 _fmin(f32 a, f32 b) noexcept nogil: return a if a < b else b
cdef inline f32 _clamp1(f32 a) noexcept nogil: return _fmax(PRO_MIN, _fmin(a, PRO_MAX))


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

# Extract and center chunk (frequencies) for randomized SVD
cpdef void memCenter(
		const u8[:,::1] G, f32[:,::1] X, f32[::1] f, const Py_ssize_t M_w
	) noexcept nogil:
	cdef:
		Py_ssize_t M = X.shape[0]
		Py_ssize_t N = X.shape[1]
		Py_ssize_t B = G.shape[1]
		size_t b, i, j, l, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 g, byte
		f32 fl
	for j in prange(M, schedule='guided'):
		i = 0
		l = M_w + j
		fl = f[l]
		for b in range(B):
			byte = G[l,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					X[j,i] = <f32>g - 2.0*fl
				else:
					X[j,i] = 0.0
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Extract and center chunk (SVD) for randomized SVD
cpdef void memCenterSVD(
		const u8[:,::1] G, f32[:,::1] U, const f32[:,::1] V, f32[:,::1] X, f32[::1] f, const Py_ssize_t M_w
	) noexcept nogil:
	cdef:
		Py_ssize_t M = X.shape[0]
		Py_ssize_t N = X.shape[1]
		Py_ssize_t B = G.shape[1]
		Py_ssize_t K = U.shape[1]
		size_t b, i, j, l, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 g, byte
		f32 fl
		f32* Ul
	for j in prange(M, schedule='guided'):
		i = 0
		l = M_w + j
		fl = f[l]
		Ul = &U[l,0]
		for b in range(B):
			byte = G[l,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					X[j,i] = <f32>g - 2.0*fl
				else:
					X[j,i] = innerE(Ul, &V[i,0], fl, K)
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Extract and standardize chunk (frequencies) for randomized SVD
cpdef void memFinal(
		const u8[:,::1] G, f32[:,::1] X, f32[::1] f, f32[::1] d, const Py_ssize_t M_w
	) noexcept nogil:
	cdef:
		Py_ssize_t M = X.shape[0]
		Py_ssize_t N = X.shape[1]
		Py_ssize_t B = G.shape[1]
		size_t b, i, j, l, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 g, byte
		f32 fl, dl
	for j in prange(M, schedule='guided'):
		i = 0
		l = M_w + j
		fl = f[l]
		dl = d[l]
		for b in range(B):
			byte = G[l,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					X[j,i] = (<f32>g - 2.0*fl)*dl
				else:
					X[j,i] = 0.0
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break

# Extract and standardize chunk (SVD) for randomized SVD
cpdef void memFinalSVD(
		const u8[:,::1] G, f32[:,::1] U, const f32[:,::1] V, f32[:,::1] X, f32[::1] f, f32[::1] d, const Py_ssize_t M_w
	) noexcept nogil:
	cdef:
		Py_ssize_t M = X.shape[0]
		Py_ssize_t N = X.shape[1]
		Py_ssize_t B = G.shape[1]
		Py_ssize_t K = U.shape[1]
		size_t b, i, j, l, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 g, byte
		f32 fl, dl
		f32* Ul
	for j in prange(M, schedule='guided'):
		i = 0
		l = M_w + j
		fl = f[l]
		dl = d[l]
		Ul = &U[l,0]
		for b in range(B):
			byte = G[l,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g != 9:
					X[j,i] = (<f32>g - 2.0*fl)*dl
				else:
					X[j,i] = innerE(Ul, &V[i,0], fl, K)*dl
				byte = byte >> 2 # Right shift 2 bits
				i = i + 1
				if i == N:
					break
