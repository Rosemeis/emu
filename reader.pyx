import os
import numpy as np
cimport numpy as np
from cython import boundscheck, wraparound
from libcpp.vector cimport vector

DTYPE = np.int8
ctypedef np.int8_t DTYPE_t

# Read file
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