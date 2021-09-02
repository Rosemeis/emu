# EMU
Version 0.72

More information, regarding options in EMU, can be found here: http://www.popgen.dk/software/index.php/EMU

## Citation
Please cite our paper in *Bioinformatics*: https://doi.org/10.1093/bioinformatics/btab027

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
Dependencies can of course also be installed with *conda*.

## Usage
### Running EMU
EMU works on PLINK files directly.
```bash
# See all options
python emu.py -h

# Using PLINK files directly (.bed, .bim, .fam) - Give prefix
python emu.py -p test -e 2 -t 64 -o test.emu
```

### Acceleration (Default - Recommended)
An acceleration scheme is being used with both SVD options (halko or arpack). Each iteration will be longer as 2 extra steps are performed but the overall number of iterations for convergence is decreased significantly. The acceleration can be turned off.
```bash
# No acceleration
python emu.py -p test -e 2 -t 64 -o test.emu.no_accel --no_accel
```

### Saving and loading factor matrices
It is recommended to save estimated factor matrices of the SVD using the parameter "-indf_save". They can then be used a start point for a new run if the results have not converged or if user wants to perform selection scan afterwards population structure inference. This saves a ton of time in the estimations.
```bash
# Saves factor matrices (test.emu.w.npy, test.emu.s.npy, test.emu.u.npy)
python emu.py -p test -e 2 -t 64 -o test.emu --indf_save

# Use factor matrices as start point
python emu.py -p test -e 2 -t 64 -o test.emu --umat test.emu.u.npy --svec test.emu.s.npy --wmat test.emu.w.npy
```

### Memory efficient implementation
A more memory efficient implementation has been added. It is based of the Halko algorithm but using custom matrix multiplications that can handle decomposed matrices. Only factor matrices as well as the 2-bit data matrix is kept in memory.
```bash
# Example run
python emu.py -m -p test -e 2 -t 64 -o test.emu.mem
```
