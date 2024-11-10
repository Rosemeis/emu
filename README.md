# EMU
EMU is a software for performing principal component analysis (PCA) in the presence of missingness for genetic datasets. EMU can handle both random and non-random missingness by modelling it directly through a truncated SVD approach. EMU uses binary PLINK files as input.

### Citation
Please cite our paper in *Bioinformatics*: https://doi.org/10.1093/bioinformatics/btab027

## Installation
```bash
# Build and install via PyPI
pip install emu-popgen

# Download source and install via pip
git clone https://github.com/Rosemeis/emu.git
cd emu
pip install .

# Download source and install in new Conda environment
git clone https://github.com/Rosemeis/emu.git
conda env create -f environment.yml
conda activate emu

# You can now run the program with the `emu` command
```

## Quick usage
### Running EMU
Provide `emu` with the file prefix of the PLINK files.
```bash
# Check help message of the program
emu -h

# Model and extract 2 eigenvectors using the EM-PCA algorithm
emu --bfile test --eig 2 --threads 64 --out test.emu
```

### Memory efficient implementation
A more memory efficient implementation has been added. It is based of the randomized SVD algorithm using custom matrix multiplications that can handle decomposed matrices. Only factor matrices as well as the 2-bit genotype matrix is kept in memory.
```bash
# Example run using '--mem' argument
emu --mem --bfile test -eig 2 -threads 64 -out test.emu.mem
```
