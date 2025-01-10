# EMU (v1.2.0)
EMU is a software for performing principal component analysis (PCA) in the presence of missingness for genetic datasets. EMU can handle both random and non-random missingness by modelling it directly through a truncated SVD approach. EMU uses binary PLINK files as input.

### Citation
Please cite our paper in [*Bioinformatics*](https://doi.org/10.1093/bioinformatics/btab027)

## Installation
```bash
# Option 1: Build and install via PyPI
pip install emu-popgen

# Option 2: Download source and install via pip
git clone https://github.com/Rosemeis/emu.git
cd emu
pip install .

# Option 3: Download source and install in a new Conda environment
git clone https://github.com/Rosemeis/emu.git
conda env create -f emu/environment.yml
conda activate emu
```
You can now run the program with the `emu` command.

## Quick usage
### Running EMU
Provide `emu` with the file prefix of the PLINK files.
```bash
# Check help message of the program
emu -h

# Model and extract 2 eigenvectors using the EM-PCA algorithm
emu --bfile test --eig 2 --threads 64 --out test.emu
```

### Memory-efficient variant
Very memory-efficient variant of `emu` for large-scale datasets.
```bash
# Example run using '--mem' argument
emu --mem --bfile test -eig 2 -threads 64 -out test.emu.mem
```
