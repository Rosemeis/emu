"""
EMU.
Main caller. Performs iterative SVD of allele count matrix (EM-PCA).

Jonas Meisner, Siyang Liu, Mingxi Huang and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import sys
from datetime import datetime
from time import time

VERSION = "1.6.0"

# Argparse
parser = argparse.ArgumentParser(prog="emu")
parser.add_argument("--version", action="version", \
	version=f"{VERSION}")
parser.add_argument("-b", "--bfile", metavar="PREFIX",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-e", "--eig", metavar="INT", type=int,
	help="Number of eigenvectors to use in iterative estimation")
parser.add_argument("-t", "--threads", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-o", "--out", metavar="OUTPUT", default="emu",
	help="Prefix output name",)
parser.add_argument("--mem", action="store_true",
	help="Memory-efficient variant")
parser.add_argument("--selection", action="store_true",
	help="Perform PC-based selection scan (Galinsky et al. 2016)")
parser.add_argument("--iter", metavar="INT", type=int, default=100,
	help="Maximum iterations in estimation of individual allele frequencies (100)")
parser.add_argument("--tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance in update for loadings (1e-5)")
parser.add_argument("--power", metavar="INT", type=int, default=11,
	help="Number of power iterations in randomized SVD (11)")
parser.add_argument("--batch", metavar="INT", type=int, default=8192,
	help="Number of SNPs to use in batches of memory variant (8192)")
parser.add_argument("--seed", metavar="INT", type=int, default=42,
	help="Set random seed")
parser.add_argument("--eig-out", metavar="INT", type=int,
	help="Number of eigenvectors to output in final SVD")
parser.add_argument("--loadings", action="store_true",
	help="Save SNP loadings")
parser.add_argument("--raw", action="store_true",
	help="Raw output without '*.fam' info")
parser.add_argument("--noise", metavar="FLOAT", type=float,
	help="Inject Gaussian noise in frequency initialization (SD)")


##### EMU main caller #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("--------------------------------")
	print(f"EMU v{VERSION}")
	print(f"Using {args.threads} thread(s).")
	print("--------------------------------\n")

	# Check input
	assert args.bfile is not None, "No input data (--bfile)"
	assert args.eig is not None, "Must specify number of eigenvectors to use!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.seed >= 0, "Please select a valid seed!"
	assert args.iter >= 0, "Please select a valid number of iterations!"
	assert args.tole >= 0.0, "Please select a valid tolerance!"
	assert args.batch > 1, "Please select a valid number of batches!"
	assert args.power > 1, "Please select a valid number of power iterations!"
	if args.eig_out is not None:
		assert args.eig_out > 1, "Please select a valid number of output eigenvectors!"
	if args.noise is not None:
		assert args.noise > 0.0, "Please provide a valid standard deviation for noise injection!"
	start = time()

	# Create log-file of arguments
	full = vars(parser.parse_args())
	deaf = vars(parser.parse_args([]))
	with open(f"{args.out}.log", "w") as log:
		log.write(f"EMU v{VERSION}\n")
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
	os.environ["MKL_MAX_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_MAX_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

	# Load numerical libraries
	import numpy as np
	from emu import functions
	from emu import shared

	# Set K
	K = args.eig if args.eig_out is None else args.eig_out

	# Read data
	assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
	print("Reading data...", end="", flush=True)
	G, M, N = functions.readPlink(args.bfile)
	print(f"\rLoaded {N} samples and {M} SNPs.")

	# Estimate allele frequencies
	n = np.zeros(M, dtype=np.float32)
	f = np.zeros(M, dtype=np.float32)
	shared.estimateF(G, f, n, N)

	# Check input
	assert (not np.allclose(np.max(f), 1.0)) or (not np.allclose(np.min(f), 0.0)), "Fixed SNPs in dataset!"
	if np.allclose(n, np.full(M, 2*N, dtype=np.float32)):
		print("No missingness in data!")
		args.iter = 0
	del n

	# Run options dictionary
	run = {
		"iter":args.iter,
		"seed":args.seed,
		"tole":args.tole,
		"batch":args.batch,
		"noise":args.noise,
		"power":args.power
	}

	# Perform EM-PCA
	print(f"\nPerforming EMU using {args.eig} eigenvector(s).")
	E = None if args.mem else np.zeros((M, N), dtype=np.float32)
	res = functions.emuAlgorithm(G, E, f, N, args.eig, K, run)
	del G, E

	# Print elapsed time for estimation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"\nTotal elapsed time: {t_min}m{t_sec}s")

	# Save matrices
	if args.raw:
		np.savetxt(f"{args.out}.eigvecs", res["V"], fmt="%.7f")
	else:
		F = np.loadtxt(f"{args.bfile}.fam", usecols=[0,1], dtype=np.str_)
		hdr = ["#FID", "IID"] + [f"PC{k}" for k in range(1, K + 1)]
		res["V"] = np.hstack((F, np.round(res["V"], 7)))
		np.savetxt(f"{args.out}.eigvecs", res["V"], fmt="%s", delimiter="\t", header="\t".join(hdr), comments="")
	np.savetxt(f"{args.out}.eigvals", (res["S"]**2)/float(M), fmt="%.7f")
	print(f"Saved eigenvector(s) as {args.out}.eigvecs")
	print(f"Saved eigenvalue(s) as {args.out}.eigvals")

	# Save SNP loadings
	if args.loadings:
		np.savetxt(f"{args.out}.loadings", res["U"], fmt="%.7f")
		print(f"Saved SNP loadings as {args.out}.loadings")

	# Perform genome-wide selection scan
	if args.selection:
		D = np.zeros((M, K), dtype=np.float32)
		shared.galinskyScan(res["U"], D)
		np.savetxt(f"{args.out}.selection", D, fmt="%.7f")
		print(f"Saved test statistics as {args.out}.selection")

	# Write output info to log-file
	with open(f"{args.out}.log", "a") as log:
		if args.iter > 0:
			if res["conv"]:
				log.write(f"\nEM-PCA converged in {res["iter"]} iterations.\n")
			else:
				log.write("\nEM-PCA did not converge!\n")
		log.write(f"Total elapsed time: {t_min}m{t_sec}s\n")
		log.write(f"Saved eigenvalue(s) as {args.out}.eigvals\n")
		log.write(f"Saved eigenvector(s) as {args.out}.eigvecs\n")
		if args.loadings:
			log.write(f"Saved SNP loadings as {args.out}.loadings\n")
		if args.selection:
			log.write(f"Saved test statistics as {args.out}.selection\n")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'emu' command!"
