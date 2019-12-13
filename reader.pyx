import os
import numpy as np
cimport numpy as np
from cython import boundscheck, wraparound
from libcpp.vector cimport vector
from libc.math cimport isnan

DTYPE = np.int8
ctypedef np.int8_t DTYPE_t

# Set seed
np.random.seed(0)

# Read PLINK file
@boundscheck(False)
@wraparound(False)
cpdef convertBed(signed char[:,::1] D, float[:,::1] G):
	cdef int n = D.shape[0]
	cdef int m = D.shape[1]
	cdef int i, j
	for i in range(n):
		for j in range(m):
			if isnan(G[i,j]): # Missing site
				D[i,j] = -9
			elif G[i,j] == 0.0:
				D[i,j] = 1
			elif G[i,j] == 1.0: # Heterozygous site
				D[i,j] = 2
			elif G[i,j] == 2.0:
				D[i,j] = 0

# Read mat file
@boundscheck(False)
@wraparound(False)
cpdef readMat(str beagle, int depth):
	cdef int c = 0
	cdef int n = 0
	cdef int i, m
	cdef signed char M = -9
	cdef list pyList
	cdef vector[vector['signed char']] D
	cdef vector['signed char'] D_ind
	with os.popen("zcat " + beagle) as f:
		for line in f:
			if c == 0:
				n = len(line.split("\t"))-6
				c += 1
				continue
			pyList = line.replace("\n", "").split("\t")
			if int(pyList[4]) > depth:
				for i in range(6, n+6):
					if pyList[i] == ".":
						D_ind.push_back(M)
					else:
						D_ind.push_back(int(pyList[i]))
				D.push_back(D_ind)
				D_ind.clear()
	m = D.size()
	cdef np.ndarray[DTYPE_t, ndim=2, mode='fortran'] D_np = np.empty((m, n), dtype=DTYPE, order='F')
	cdef signed char *D_ptr
	for i in range(m):
		D_ptr = &D[i][0]
		D_np[i] = np.asarray(<signed char[:n]> D_ptr)
	return D_np