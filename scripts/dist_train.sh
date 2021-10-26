#!/bin/bash

ROOT=../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
python -m eod train \
  --ng=$1 \
  --launch=pytorch \
  --config=$cfg \
  --display=10 \
  2>&1 | tee log.train.$T.$(basename $cfg) 
