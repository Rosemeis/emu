# EMU
Version 0.5

## Get EMU and build
Clone repository and build (It is assumed that OpenMP is installed).
```bash
git clone https://github.com/Rosemeis/emu.git
cd emu/

# Install library dependencies
pip install --user -r requirements.txt

# Build
python setup.py build_ext --inplace
```

## Usage
### Convert binary PLINK files into .npy. We rely on the pandas-plink (v. 1.2.31) to read PLINK files in the current EMU version. We are working on a custom PLINK reader for future versions.
```bash
# Give the PLINK file prefix
python convertMat.py -plink test -o test
```

### Convert binary PLINK .bed file into normal binary (8-bit signed char)
```bash
# Give the PLINK file prefix
python convertMat.py -plink test -binary -o test
``` 

### Convert binary PLINK .bed file into normal binary in blocks (8-bit signed char)
```bash
# Give the PLINK file prefix
python convertMat.py -plink test -binary -block -block_size 4096 -o test
``` 

EMU can therefore read any binary genotype matrix file given that the entries are stored in signed chars.

### Convert matrix into .npy (Deprecated! Old custom format - Legacy)
```bash
# Without generating index vector for guidance
python convertMat.py -mat test.mat.gz -o test

# Generating index vector for guidance (using .ped file)
python convertMat.py -mat test.mat.gz -ped test.ped -o test
```

### Running EMU
```bash
# Help
python emu.py -h

# Without guidance
python emu.py -npy test.npy -e 2 -t 64 -o test.emu

# With guidance
python emu.py -npy test.npy -index test.index.npy -e 2 -t 64 -o test.emu

# Using binary file (.bin) - need also to include number of individuals (-ind)
python emu.py -bin test.bin -ind 100 -index test.index.npy -e 2 -t 64 -o test.emu
```

### Acceleration (Recommended)
An acceleration scheme can now be used with both SVD options (arpack or halko) using "-accel". Each iteration will be longer as 3 steps are performed but the overall number of iterations for convergence is decreased significantly.
```bash
# Acceleration with Halko
python emu.py -npy test.npy -index test.index.npy -e 2 -t 64 -svd halko -accel -o test.emu.accel
```

### Saving and loading factor matrices
It is recommended to save estimated factor matrices of the SVD using the parameter "-indf_save". They can then be used a start point for a new run if the results have not converged or if user wants to perform selection scan afterwards population structure inference. This saves a ton of time in the estimations.
```bash
# Saves factor matrices (test.emu.w.npy, test.emu.s.npy, test.emu.u.npy)
python emu.py -npy test.npy -index test.index.npy -e 2 -t 64 -o test.emu -indf_save

# Use factor matrices as start point and performing selection scan immediately
python emu.py -npy test.npy -index test.index.npy -e 2 -t 64 -o test.emu -w test.emu.w.npy -s test.emu.s.npy -u test.emu.u.npy -selection -m 0
```

### Memory efficient implementation
A more memory efficient implementation has been added. It is based of the Halko algorithm but using custom matrix multiplications that can handle decomposed matrices.
```bash
# Example run
python emu_mem.py -npy test.npy -e 2 -t 64 -accel -o test.memory.emu.accel
``` 