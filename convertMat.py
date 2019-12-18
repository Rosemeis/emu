"""
Converts PLINK format into NumPy binary format for EMU.

Example usage: python convertMat.py -plink emu_plink -o emu_mat
"""

__author__ = "Jonas Meisner"

# Libraries
import reader
import numpy as np
import argparse
import subprocess


# Find length of PLINK files
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, err = process.communicate()
	return int(result.split()[0])

##### Argparse #####
parser = argparse.ArgumentParser()
parser.add_argument("-plink", metavar="FILE-PREFIX",
	help="Prefix for binary PLINK files (.bed, .bim, .fam)")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="matrix")
args = parser.parse_args()

### Main ###
print("Reading in single-read sampling matrix from PLINK files.")
assert args.plink is not None, "No valid input given! Must use '-plink'!"
# Finding length of .fam and .bim file and read .bed file into NumPy array
n = extract_length(args.plink + ".fam")
m = extract_length(args.plink + ".bim")
print("Matrix size: (" + str(n) + " x " + str(m) + ")")

D = np.zeros((n, m), dtype=np.int8)
reader.readBed(args.plink + ".bed", D, n, m)

print("Saving matrix in binary Numpy format (.npy)")
np.save(args.o, D)