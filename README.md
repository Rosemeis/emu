# EMU

[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?colorB=58bd9f&style=popout)](http://bioconda.github.io/recipes/emu-pca/README.html)

EMU is a software for performing principal component analysis (PCA) in the presence of missingness for genetic datasets. EMU can handle both random and non-random missingness by modelling it directly through a truncated SVD approach. EMU uses binary PLINK files as input.

### Citation
Please cite our paper in *Bioinformatics*: https://doi.org/10.1093/bioinformatics/btab027

### Dependencies
The EMU software relies on the following two Python packages that you can install through conda (recommended) or pip:

- numpy
- cython

You can create an environment through conda easily as follows:
```bash
conda env create -f environment.yml
```

## Install and build
```bash
git clone https://github.com/Rosemeis/emu.git
cd emu
pip3 install .
```

You can now run EMU with the `emu` command.

## Usage
### Running EMU
EMU works directly on PLINK files.
```bash
# See all options
emu -h

# Using PLINK files directly (test.bed, test.bim, test.fam) - Give prefix
emu --bfile test --n_eig 2 --threads 64 --out test.emu
```

### Memory efficient implementation
A more memory efficient implementation has been added. It is based of the randomized SVD algorithm ([Halko et al.](https://arxiv.org/abs/0909.4061)) but using custom matrix multiplications that can handle decomposed matrices. Only factor matrices as well as the 2-bit data matrix is kept in memory.
```bash
# Example run using '-m' argument
emu --mem --bfile test -e 2 -t 64 -o test.emu.mem
```
