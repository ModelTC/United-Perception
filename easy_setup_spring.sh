#!/bin/bash

export PATH=/mnt/lustre/share/gcc/gcc-5.3.0/bin/:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gmp-4.3.2/lib:/mnt/lustre/share/gcc/mpfr-2.4.2/lib:/mnt/lustre/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST='3.5;5.0+PTX;6.0;7.0'

pip uninstall -y nart_tools
pip uninstall -y nart==0.2.4
# pip uninstall -y torchvision==0.4.2
pip uninstall -y springvision==1.0.1
pip uninstall -y kestrel==1.5.4-patch1
pip install --user -r requirements.txt

partition=$1
spring.submit run -p $partition -n1 --gpu "python setup.py build_ext -i"
# python setup.py build_ext -i
