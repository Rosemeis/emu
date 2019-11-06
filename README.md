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
### Convert binary PLINK files into .npy
```bash
# Give the PLINK file prefix
python convertMat.py -plink test -o test
```

### Convert matrix into .npy (Deprecated! Old custom format)
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
python emu.py test.npy -e 2 -t 64 -o test.emu

# With guidance
python emu.py test.npy -index test.index.npy -e 2 -t 64 -o test.emu
```

### Acceleration (Recommended)
An acceleration scheme can now be used with both SVD options (arpack or halko) using "-accel". Each iteration will be longer as 3 steps are performed but the overall number of iterations for convergence is decreased significantly.
```bash
# Acceleration with Halko
python emu.py test.npy -index test.index.npy -e 2 -t 64 -svd halko -accel -o test.emu.accel
```

### Saving and loading factor matrices
It is recommended to save estimated factor matrices of the SVD using the parameter "-indf_save". They can then be used a start point for a new run if the results have not converged or if user wants to perform selection scan afterwards population structure inference. This saves a ton of time in the estimations.
```bash
# Saves factor matrices (test.emu.w.npy, test.emu.s.npy, test.emu.u.npy)
python emu.py test.npy -index test.index.npy -e 2 -t 64 -o test.emu -indf_save

# Use factor matrices as start point and performing selection scan immediately
python emu.py test.npy -index test.index.npy -e 2 -t 64 -o test.emu -w test.emu.w.npy -s test.emu.s.npy -u test.emu.u.npy -selection -m 0
```

### Memory efficient implementation
A more memory efficient implementation has been added. It is based of the Halko algorithm but using custom matrix multiplications that can handle decomposed matrices. Due to the requirement of parallelization (speed), user also needs to specify the transposed 8-bit integer matrix in C-contiguous memory. Therefore the memory requirement of this implementation will be around, 2xMxN bytes. It is also assumed that MAF filtering has been performed prior to running this method.
```bash
# Create and save transposed matrix in C-contiguous memory
python -c "import numpy as np; D=np.load('test.npy'); np.save('test.trans.npy', np.ascontiguousarray(D.T, dtype=np.int8))"

# Example run
python emu_mem.py -D test.npy -Dt test.trans.npy -e 2 -t 64 -accel -o test.memory.emu.accel
``` 