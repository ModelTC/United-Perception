#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
jobname=flops

srun -N$1 --gres=gpu:$2 -p $3 --job-name=$jobname --cpus-per-task=2 \
python -m up flops \
   --config=$cfg \
   --input_size=3,224,224 \
   2>&1 | tee log.flops.$T.$(basename, $cfg)
