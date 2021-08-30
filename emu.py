"""
EMU.
Main caller. Performs iterative SVD of allele count matrix (EM-PCA) based on either ARPACK, or Halko method.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen

Example usages:
```
# EMU
python emu.py -plink fileprefix -e 2 -threads 64 -out emu

# EMU-mem
python emu.py -plink fileprefix -mem -e 2 -threads 64 -out emu_mem
```
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import subprocess

##### Argparse #####
parser = argparse.ArgumentParser(prog="EMU")
parser.add_argument("--version", action="version", version="%(prog)s alpha 0.72")
parser.add_argument("-mem", action="store_true",
	help="EMU-mem variant")
parser.add_argument("-plink", metavar="FILE-PREFIX",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-e", metavar="INT", type=int,
	help="Number of eigenvectors to use in iterative estimation")
parser.add_argument("-k", metavar="INT", type=int,
	help="Number of eigenvectors to output in final SVD")
parser.add_argument("-m", metavar="INT", type=int, default=100,
	help="Maximum iterations in estimation of individual allele frequencies (100)")
parser.add_argument("-m_tole", metavar="FLOAT", type=float, default=5e-7,
	help="Tolerance in update for individual allele frequencies (5e-7)")
parser.add_argument("-threads", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-maf", metavar="FLOAT", type=float, default=0.00,
	help="Threshold for minor allele frequencies (0.00)")
parser.add_argument("-selection", action="store_true",
	help="Perform PC-based selection scan (Galinsky et al. 2016)")
parser.add_argument("-maf_save", action="store_true",
	help="Save estimated population allele frequencies")
parser.add_argument("-sites_save", action="store_true",
	help="Save vector of sites after MAF filtering (Binary)")
parser.add_argument("-indf_save", action="store_true",
	help="Save estimated singular matrices")
parser.add_argument("-svd", metavar="STRING", default="arpack",
	help="Method for performing truncated SVD (arpack/halko)")
parser.add_argument("-svd_power", metavar="INT", type=int, default=4,
	help="Number of power iterations in randomized SVD (Halko)")
parser.add_argument("-u", metavar="FILE",
	help="left singular matrix (.u.npy)")
parser.add_argument("-s", metavar="FILE",
	help="Singular values (.s.npy)")
parser.add_argument("-w", metavar="FILE",
	help="right singular matrix (.w.npy)")
parser.add_argument("-no_accel", action="store_true",
	help="Turn off acceleration for EM (not recommended)")
parser.add_argument("-cost", action="store_true",
	help="Output min-cost each iteration (ONLY EMU)")
parser.add_argument("-cost_step", action="store_true",
	help="Use acceleration based on cost (ONLY EMU)")
parser.add_argument("-out", metavar="OUTPUT", default="emu",
	help="Prefix output name",)
args = parser.parse_args()

### Caller ###
print("EMU 0.72\n")

# Control threads
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
os.environ["MKL_NUM_THREADS"] = str(args.threads)

# Import numerical libraries
import numpy as np
from math import ceil

# Import own scripts
import shared
import shared_cy

# Set K
assert args.e is not None, "Must specify number of eigenvectors to use!"
if args.k is None:
	K = args.e
else:
	K = args.k

# Reader help function
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, err = process.communicate()
	return int(result.split()[0])

# Read data
print("Reading in data matrix from PLINK files.")
assert args.plink is not None, "No valid input given! Must use '-plink'!"
# Finding length of .fam and .bim file and read .bed file into NumPy array
n = extract_length(args.plink + ".fam")
m = extract_length(args.plink + ".bim")
bed = open(args.plink + ".bed", "rb")
B = np.fromfile(bed, dtype=np.uint8, offset=3)
bed.close()
Bi = ceil(n/4) # Length of bytes to describe n individuals
D = B.reshape((m, Bi))
m_old = D.shape[0] # For future reference
print("Loaded " + str(n) + " samples and " + str(m) + " sites.")

# Population allele frequencies
print("Estimating population allele frequencies.")
f = np.zeros(m, dtype=np.float32)
shared_cy.estimateF(D, f, Bi, n, m, args.threads)

# Removing rare variants
if args.maf > 0.0:
	mask = (f >= args.maf) & (f <= (1 - args.maf))

	# Filter and update arrays without copying
	m = np.sum(mask)
	tmpMask = mask.astype(np.uint8)
	shared_cy.filterArrays(D, f, tmpMask)
	D = D[:m,:]
	f = f[:m]
	del tmpMask
	print("Number of sites after MAF filtering (" + str(args.maf) + "): " \
            + str(m))
	m = D.shape[0]

# Use eigenvectors from previous run
if args.w is not None:
	assert args.s is not None, "Must supply both -s and -u along with -w!"
	assert args.u is not None, "Must supply both -s and -u along with -w!"
	print("Reading singular matrices from previous run.")
	W = np.load(args.w)
	assert W.shape[1] == n, "Number of samples in W must match D!"
	s = np.load(args.s)
	U = np.load(args.u)
	assert U.shape[0] == m, "Number of sites in U must match D!"
else:
	W, s, U = None, None, None

# Additional parsing options
if args.cost_step:
	assert args.cost, "Must also estimate cost at every iteration (-cost)!"
if args.no_accel:
	print("Turned off EM acceleration.")
	accel = False
else:
	accel = True

##### EMU #####
if args.mem:
	print("\nPerforming EMU-mem using " + str(args.e) + " eigenvector(s).")
	U, s, V = shared.emuMemory(D, f, args.e, K, args.m, args.m_tole, U, s, W, \
								Bi, n, m, args.svd_power, args.indf_save, \
								args.out, accel, args.threads)
else:
	print("\nPerforming EMU using " + str(args.e) + " eigenvector(s).")
	U, s, V = shared.emuAlgorithm(D, f, args.e, K, args.m, args.m_tole, U, s, \
									W, Bi, n, m, args.svd, args.svd_power, \
									args.indf_save, args.out, accel, args.cost,\
									args.cost_step, args.threads)

# Save matrices
np.savetxt(args.out + ".eigenvecs", V.T, fmt="%.7f")
print("Saved eigenvector(s) as " + args.out + ".eigenvecs (Text).")
np.savetxt(args.out + ".eigenvals", s**2/float(m), fmt="%.7f")
print("Saved eigenvalue(s) as " + args.out + ".eigenvals (Text).")
del V, s

# Perform genome-wide selection scan
if args.selection:
	print("\nPerforming genome-wide selection scan along each PC.")
	Dsquared = np.zeros((m, K), dtype=np.float32)
	shared_cy.galinskyScan(U, Dsquared)
	np.save(args.out + ".selection", Dsquared.astype(float, copy=False))
	print("Saved test statistics as " + args.out + ".selection.npy (Binary).")
	del Dsquared
del U

# Optional saves
if args.maf_save:
	np.save(args.out + ".maf", f.astype(float, copy=False))
	print("\nSaved minor allele frequencies as " + args.out + ".maf.npy (Binary).")
del f
if (args.sites_save) and (args.maf > 0.0):
	siteVec = np.zeros(m_old, dtype=np.uint8)
	siteVec[mask] = 1
	np.savetxt(args.out + ".sites", siteVec, fmt="%i")
	print("\nSaved boolean vector of sites kept after filtering as " + \
			str(args.out) + ".sites (Text)")
