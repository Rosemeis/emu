# FlashPCAngsd

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
