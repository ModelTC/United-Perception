#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$1
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

spring.submit run -n1 -p spring_scheduler --gpu --job-name=$2 --cpus-per-task=${CPUS_PER_TASK} \
"python -m eod inference \
  --config=$cfg \
  2>&1 | tee log.inference.$T.$(basename $cfg) "
