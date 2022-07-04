#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
jobname=test


srun -N$1 --gres=gpu:$2 -p $3 --job-name=$jobname --cpus-per-task=2 \
python -m up train \
  --ng=$2 \
  --launch=pytorch \
  -e \
  --config=$cfg \
  --display=10 \
  2>&1 | tee log.test.$T.$(basename $cfg)
