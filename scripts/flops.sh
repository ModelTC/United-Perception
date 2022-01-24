#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

spring.submit run -n$1 -p spring_scheduler --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
"python -m up flops \
   --config=$cfg \
   --input_size=3,224,224 \
   2>&1 | tee log.flops.$T.$(basename, $cfg) "
