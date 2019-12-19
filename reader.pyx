import os
import numpy as np
cimport numpy as np
from cython import boundscheck, wraparound
from libc.stdio cimport fopen, fclose, FILE, fread

# Read .bed file
@boundscheck(False)
@wraparound(False)
cpdef readBed(str bedfile, signed char[:,::1] D, int n, int m):
	cdef signed char[4] recode = [1, -9, 2, 0] # EMU format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef unsigned char start[3]
	cdef int bytepart = 0, i = 0, j = 0
	cdef FILE *bed = fopen(bedfile.encode('utf-8'), "r")
	fread(start, 1, 3, bed) # Read first three bytes - 0x6c, 0x1b, and 0x01

	while True:
		if bytepart == 0: # Read byte
			fread(&byte, 1, 1, bed)
			bytepart = 4
		code = byte & mask
		D[i,j] = recode[code]
		byte = byte >> 2
		bytepart -= 1
		i += 1
		if i == n:
			i = 0
			bytepart = 0
			j += 1
			if j == m:
				break
	fclose(bed)