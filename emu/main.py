"""
EMU.
Main caller. Performs iterative SVD of allele count matrix (EM-PCA) based on either ARPACK, or Halko method.

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import sys
from datetime import datetime
from time import time

# Argparse
parser = argparse.ArgumentParser(prog="emu")
parser.add_argument("--version", action="version", version="%(prog)s 1.0")
parser.add_argument("-b", "--bfile", metavar="FILE-PREFIX",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-e", "--n_eig", metavar="INT", type=int,
	help="Number of eigenvectors to use in iterative estimation")
parser.add_argument("-k", "--n_out", metavar="INT", type=int,
	help="Number of eigenvectors to output in final SVD")
parser.add_argument("-t", "--threads", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-f", "--maf", metavar="FLOAT", type=float,
	help="Threshold for minor allele frequencies")
parser.add_argument("-s", "--selection", action="store_true",
	help="Perform PC-based selection scan (Galinsky et al. 2016)")
parser.add_argument("-o", "--out", metavar="OUTPUT", default="emu",
	help="Prefix output name",)
parser.add_argument("-m", "--mem", action="store_true",
	help="EMU-mem variant")
parser.add_argument("--iter", metavar="INT", type=int, default=100,
	help="Maximum iterations in estimation of individual allele frequencies (100)")
parser.add_argument("--tole", metavar="FLOAT", type=float, default=5e-7,
	help="Tolerance in update for individual allele frequencies (5e-7)")
parser.add_argument("--power", metavar="INT", type=int, default=11,
	help="Number of power iterations in randomized SVD (11)")
parser.add_argument("--batch", metavar="INT", type=int, default=4096,
	help="Number of SNPs to use in batches of memory variant (4096)")
parser.add_argument("--loadings", action="store_true",
	help="Save SNP loadings")
parser.add_argument("--maf_save", action="store_true",
	help="Save estimated population allele frequencies")
parser.add_argument("--cost", action="store_true",
	help="Output min-cost each iteration (DEBUG function)")
parser.add_argument("--seed", metavar="INT", type=int, default=0,
	help="Set random seed")


##### EMU main caller #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("--------------------------------")
	print("EMU v1.0")
	print(f"Using {args.threads} thread(s).")
	print("--------------------------------\n")

	# Check input
	assert args.bfile is not None, "No input data (--bfile)"
	assert args.n_eig is not None, "Must specify number of eigenvectors to use!"
	start = time()

	# Create log-file of arguments
	full = vars(parser.parse_args())
	deaf = vars(parser.parse_args([]))
	with open(f"{args.out}.log", "w") as log:
		log.write("EMU v1.0\n")
		log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
		log.write(f"Directory: {os.getcwd()}\n")
		log.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				if type(full[key]) is bool:
					log.write(f"\t--{key}\n")
				else:
					log.write(f"\t--{key} {full[key]}\n")
	del full, deaf

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Load numerical libraries
	import numpy as np
	from math import ceil
	from emu import algorithm
	from emu import memory
	from emu import shared
	from emu import shared_cy

	# Set K
	if args.n_out is None:
		K = args.n_eig
	else:
		K = args.n_out

	# Read data
	print("Reading in data matrix from PLINK files.")
	assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
	N = shared.extract_length(f"{args.bfile}.fam")
	M = shared.extract_length(f"{args.bfile}.bim")
	with open(f"{args.bfile}.bed", "rb") as bed:
		D = np.fromfile(bed, dtype=np.uint8, offset=3)
	B = ceil(N/4) # Length of bytes to describe n individuals
	D.shape = (M, B)
	print(f"Loaded {N} samples and {M} SNPs.", flush=True)

	# Population allele frequencies
	print("Estimating population allele frequencies.")
	f = np.zeros(M, dtype=np.float32)
	shared_cy.estimateF(D, f, N, args.threads)

	# Removing rare variants
	if args.maf is not None:
		assert (args.maf > 0.0) and (args.maf < 1.0), "Please provide a valid MAF!"
		mask = (f >= args.maf) & (f <= (1.0 - args.maf))

		# Filter and update arrays without copying
		M = np.sum(mask)
		tmpMask = mask.astype(np.uint8)
		shared_cy.filterArrays(D, f, tmpMask, N)
		D = D[:M,:]
		f = f[:M]
		del tmpMask
		print(f"Number of sites after MAF filtering ({args.maf}): {M}")
		M = D.shape[0]
	assert (not np.allclose(np.max(f), 1.0)) or (not np.allclose(np.min(f), 0.0)), \
		"Fixed sites in dataset. Must perform MAF filtering (-f / --maf)!"

	##### EMU #####
	if args.mem:
		print(f"\nPerforming EMU-mem using {args.n_eig} eigenvector(s).")
		U, S, V, it, converged = memory.emuMemory(D, f, args.n_eig, K, N, args.iter, \
			args.tole, args.power, args.cost, args.batch, args.seed, args.threads)
	else:
		print(f"\nPerforming EMU using {args.n_eig} eigenvector(s).")
		U, S, V, it, converged = algorithm.emuAlgorithm(D, f, args.n_eig, K, N, \
			args.iter, args.tole, args.power, args.cost, args.seed, args.threads)
	del D

	# Print elapsed time for estimation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	# Save matrices
	np.savetxt(f"{args.out}.eigenvecs", V, fmt="%.7f")
	print(f"Saved eigenvector(s) as {args.out}.eigenvecs")
	np.savetxt(f"{args.out}.eigenvals", (S**2)/float(M), fmt="%.7f")
	print(f"Saved eigenvalue(s) as {args.out}.eigenvals")
	del V, S

	# Save loadings
	if args.loadings:
		np.savetxt(f"{args.out}.loadings", U, fmt="%.7f")
		print(f"Saved SNP loadings as {args.out}.loadings")

	# Perform genome-wide selection scan
	if args.selection:
		Dsquared = np.zeros((M, K), dtype=np.float32)
		shared_cy.galinskyScan(U, Dsquared)
		np.savetxt(f"{args.out}.selection", Dsquared, fmt="%.7f")
		print(f"Saved test statistics as {args.out}.selection")
		del Dsquared
	del U

	# Optional saves
	if args.maf_save:
		np.savetxt(f"{args.out}.freq", f, fmt="%.7f")
		print(f"Saved minor allele frequencies as {args.out}.freq")
	del f

	# Write output info to log-file
	with open(f"{args.out}.log", "a") as log:
		if converged:
			log.write(f"\nEM-PCA converged in {it} iterations.\n")
		else:
			log.write("\nEM-PCA did not converge!\n")
		log.write(f"Total elapsed time: {t_min}m{t_sec}s\n")
		log.write(f"Saved eigenvector(s) as {args.out}.eigenvecs\n")
		log.write(f"Saved eigenvalue(s) as {args.out}.eigenvals\n")
		if args.loadings:
			log.write(f"Saved SNP loadings as {args.out}.loadings\n")
		if args.selection:
			log.write(f"Saved test statistics as {args.out}.selection\n")
		if args.maf_save:
			log.write(f"Saved minor allele frequencies as {args.out}.freq\n")



##### Define main #####
if __name__ == "__main__":
	main()
