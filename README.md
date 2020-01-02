# EMU
Version 0.65

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
### Running EMU
EMU can work on PLINK files directly or load data from a NumPy array (np.int8 - signed char).
```bash
# See all options
python emu.py -h

# Using binary NumPy file (.npy)
python emu.py -npy test.npy -e 2 -t 64 -o test.emu

# Using PLINK files directly (.bed, .bim, .fam) - Give prefix
python emu.py -plink test -e 2 -t 64 -o test.emu

# With guidance
python emu.py -npy test.npy -index test.index.npy -e 2 -t 64 -o test.emu
```

### Convert binary PLINK files into .npy.
```bash
# Give PLINK prefix
python convertMat.py -plink test -o test
```

### Acceleration (Recommended)
An acceleration scheme can now be used with both SVD options (arpack or halko) using "-accel". Each iteration will be longer as 3 steps are performed but the overall number of iterations for convergence is decreased significantly.
```bash
# Acceleration with Halko
python emu.py -npy test.npy -index test.index.npy -e 2 -t 64 -accel -o test.emu.accel
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
A more memory efficient implementation has been added. It is based of the Halko algorithm but using custom matrix multiplications that can handle decomposed matrices. Can only read PLINK files as it uses the same 2-bit format of the .bed file.
```bash
# Example run
python emu_mem.py -plink test -e 2 -t 64 -accel -o test.memory.emu.accel
``` 