#!/bin/bash

export PATH=/your/path/to/gcc-5.3.0/bin/:$PATH # gcc path
export LD_LIBRARY_PATH=/your/path/to/gmp-4.3.2/lib/:/your/path/to/mpfr-2.4.2/lib/:/your/path/to/mpc-0.8.1/lib/:$LD_LIBRARY_PATH # lib path
export TORCH_CUDA_ARCH_LIST='3.5;5.0+PTX;6.0;7.0' # cuda list

pip install --user -r requirements.txt

python setup.py build_ext -i
