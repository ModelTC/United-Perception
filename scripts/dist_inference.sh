#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2

python -m eod inference \
  --ng=$1
  --launch=pytorch \
  --config=$cfg \
  2>&1 | tee log.inference.$T.$(basename $cfg) 
