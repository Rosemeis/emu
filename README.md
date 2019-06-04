# FlashPCAngsd
Version 0.45

## Get FlashPCAnsgd and build
```
git clone https://github.com/Rosemeis/flashpcangsd.git
cd flashpcangsd/
python setup.py build_ext --inplace
```

## Usage
### Convert matrix into .npy
```bash
# Without generating index vector for guidance
python convertMat.py test.mat.gz -o test

# Generating index vector for guidance (using .ped file)
python convertMat.py test.mat.gz -ped test.ped -o test
```

### Running FlashPCAngsd
```bash
# Help
python flashpcangsd.py -h

# Without guidance
python flashpcangsd.py test.npy -e 2 -t 64 -o test.flash

# With guidance
python flashpcangsd.py test.npy -index test.index.npy -e 2 -t 64 -o test.flash
```

### Acceleration
An acceleration scheme can now be used with both SVD options (arpack or halko) using "-accel". Each iteration will be longer as 3 steps are performed but the overall number of iterations for convergence is decreased significantly.
```bash
# Acceleration with Halko
python flashpcangsd.py test.npy -index test.index.npy -e 2 -t 64 -svd halko -accel -o test.flash.accel
```

### Saving and loading factor matrices
It is recommended to save estimated factor matrices of the SVD using the parameter "-indf_save". They can then be used a start point for a new run if the results have not converged or if user wants to perform selection scan afterwards population structure inference. This saves a ton of time in the estimations.
```bash
# Saves factor matrices (test.flash.w.npy, test.flash.s.npy, test.flash.u.npy)
python flashpcangsd.py test.npy -index test.index.npy -e 2 -t 64 -o test.flash -indf_save

# Use factor matrices as start point and performing selection scan immediately
python flashpcangsd.py test.npy -index test.index.npy -e 2 -t 64 -o test.flash -w test.flash.w.npy -s test.flash.s.npy -u test.flash.u.npy -selection -m 0
```