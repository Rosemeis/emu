"""
EMU.
Main caller. Performs iterative SVD of allele count matrix (EM-PCA) based on either ARPACK, or Halko method.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import subprocess
import sys
from datetime import datetime

# Reader help function
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, err = process.communicate()
	return int(result.split()[0])

# Argparse
parser = argparse.ArgumentParser(prog="EMU")
parser.add_argument("--version", action="version", version="%(prog)s alpha 0.75")
parser.add_argument("-m", "--mem", action="store_true",
	help="EMU-mem variant")
parser.add_argument("-p", "--plink", metavar="FILE-PREFIX",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-e", "--n_eig", metavar="INT", type=int,
	help="Number of eigenvectors to use in iterative estimation")
parser.add_argument("-k", "--n_out", metavar="INT", type=int,
	help="Number of eigenvectors to output in final SVD")
parser.add_argument("-t", "--threads", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-f", "--maf", metavar="FLOAT", type=float, default=0.00,
	help="Threshold for minor allele frequencies (0.00)")
parser.add_argument("-s", "--selection", action="store_true",
	help="Perform PC-based selection scan (Galinsky et al. 2016)")
parser.add_argument("-o", "--out", metavar="OUTPUT", default="emu",
	help="Prefix output name",)
parser.add_argument("--no_accel", action="store_true",
	help="Turn off acceleration for EM (not recommended)")
parser.add_argument("--iter", metavar="INT", type=int, default=100,
	help="Maximum iterations in estimation of individual allele frequencies (100)")
parser.add_argument("--tole", metavar="FLOAT", type=float, default=5e-7,
	help="Tolerance in update for individual allele frequencies (5e-7)")
parser.add_argument("--svd", metavar="STRING", default="halko",
	help="Method for performing truncated SVD (arpack/halko)")
parser.add_argument("--svd_power", metavar="INT", type=int, default=4,
	help="Number of power iterations in randomized SVD (Halko)")
parser.add_argument("--loadings", action="store-true",
	hep="Save SNP loadings")
parser.add_argument("--maf_save", action="store_true",
	help="Save estimated population allele frequencies")
parser.add_argument("--sites_save", action="store_true",
	help="Save vector of sites after MAF filtering (Binary)")
parser.add_argument("--cost", action="store_true",
	help="Output min-cost each iteration (ONLY EMU)")
parser.add_argument("--cost_step", action="store_true",
	help="Use acceleration based on cost (ONLY EMU)")


##### EMU #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("EMU v.0.75\n")
	assert args.plink is not None, "No input data (-plink)"

	# Create log-file of arguments
	full = vars(parser.parse_args())
	deaf = vars(parser.parse_args([]))
	with open(args.out + ".args", "w") as f:
		f.write("EMU v.0.75\n")
		f.write("Time: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
		f.write("Directory: " + str(os.getcwd()) + "\n")
		f.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				if type(full[key]) is bool:
					f.write("\t--" + str(key) + "\n")
				else:
					f.write("\t--" + str(key) + " " + str(full[key]) + "\n")
	del full, deaf

	# Control threads
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
	os.environ["MKL_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries
	import numpy as np
	from math import ceil

	# Import own scripts
	from emu import shared
	from emu import shared_cy

	# Set K
	assert args.n_eig is not None, "Must specify number of eigenvectors to use!"
	if args.n_out is None:
		K = args.n_eig
	else:
		K = args.n_out

	# Read data
	print("Reading in data matrix from PLINK files.")
	# Finding length of .fam and .bim file and read .bed file into NumPy array
	n = extract_length(args.plink + ".fam")
	m = extract_length(args.plink + ".bim")
	with open(args.plink + ".bed", "rb") as bed:
		B = np.fromfile(bed, dtype=np.uint8, offset=3)
	Bi = ceil(n/4) # Length of bytes to describe n individuals
	D = B.reshape((m, Bi))
	m_old = D.shape[0] # For future reference
	print("Loaded " + str(n) + " samples and " + str(m) + " sites.")

	# Population allele frequencies
	print("Estimating population allele frequencies.")
	f = np.zeros(m, dtype=np.float32)
	c = np.zeros(m, dtype=np.int32)
	shared_cy.estimateF(D, f, c, Bi, n, m, args.threads)
	del c

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
	assert (not np.allclose(np.max(f), 1.0)) or (not np.allclose(np.min(f), 0.0)), \
			"Fixed sites in dataset. Must perform MAF filtering (-f / --maf)!"

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
		print("\nPerforming EMU-mem using " + str(args.n_eig) + " eigenvector(s).")
		U, s, V = shared.emuMemory(D, f, args.n_eig, K, args.iter, args.tole, \
									Bi, n, m, args.svd_power, args.out, accel, \
									args.threads)
	else:
		print("\nPerforming EMU using " + str(args.n_eig) + " eigenvector(s).")
		U, s, V = shared.emuAlgorithm(D, f, args.n_eig, K, args.iter, args.tole, \
										Bi, n, m, args.svd, args.svd_power, \
										args.out, accel, args.cost, \
										args.cost_step, args.threads)

	# Save matrices
	np.savetxt(args.out + ".eigenvecs", V.T, fmt="%.7f")
	print("Saved eigenvector(s) as " + args.out + ".eigenvecs (Text).")
	np.savetxt(args.out + ".eigenvals", s**2/float(m), fmt="%.7f")
	print("Saved eigenvalue(s) as " + args.out + ".eigenvals (Text).")
	del V, s

	# Save loadings
	if args.loadings:
		np.save(args.out + ".loadings", U)
		print("Saved SNP loadings as " +  args.out + ".loadings (Binary).")

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


##### Define main #####
if __name__ == "__main__":
	main()
