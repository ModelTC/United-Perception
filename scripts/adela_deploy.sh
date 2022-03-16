#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$1
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

spring.submit run -n1 -p Test --gpu --job-name=$2 --cpus-per-task=${CPUS_PER_TASK} \
"python -m up adela_deploy \
  --config=$cfg \
  --release_json=$2 \
  2>&1 | tee log.adela_deploy.$T.$(basename $cfg) "