#!/bin/bash

# cmd example: sh train_qat.sh 1 8 ToolChain

UP=/path to up
MQB=/path to mqbench

cfg=$UP/configs/quant/det/retinanet/retinanet-r18-improve_quant_trt_qat.yaml


jobname=quant_qat


export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH


srun -N$1 --gres=gpu:$2 -p $3 --job-name=$jobname --cpus-per-task=2 \
nohup python -u -m up train \
  --ng=$2 \
  --launch=pytorch \
  --config=$cfg \
  --display=10 \
  > train_log_qat.txt 2>&1 &