#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

spring.submit run -n$1 -p spring_scheduler --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
"python -m up to_caffe \
  --config=$cfg \
  --save_prefix=tocaffe \
  --input_size=3x512x512 \
  2>&1 | tee log.tocaffe.$T.$(basename $cfg) "