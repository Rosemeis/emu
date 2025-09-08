# EMU (v1.3.0)
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

If you run into issues with your installation on a HPC system, it could be due to a mismatch of CPU architectures between login and compute nodes (illegal instruction). You can try and remove every instance of the `march=native` compiler flag in the [setup.py](./setup.py) file which optimizes `emu` to your specific hardware setup. Another alternative is to use the [uv package manager](https://docs.astral.sh/uv/), where you can run `emu` in a temporary and isolated environment by simply adding `uvx` in front of the `emu` command.

```bash
# uv tool run example
uvx emu --bfile test --eig 2 --threads 64 --out test.emu
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

### Memory-efficient variant
Very memory-efficient variant of `emu` for large-scale datasets.
```bash
# Example run using '--mem' argument
emu --mem --bfile test -eig 2 -threads 64 -out test.emu.mem
```
