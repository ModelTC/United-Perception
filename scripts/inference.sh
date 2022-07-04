#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$1
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
jobname=inference

srun -N$1 --gres=gpu:$2 -p $3 --job-name=$jobname --cpus-per-task=2 \
python -m up inference \
  --ng=$2 \
  --launch=pytorch \
  --config=$cfg \
  -i=imgs \
  -c=ckpt \
  2>&1 | tee log.inference.$T.$(basename $cfg)
