#!/bin/bash

# cmd example: sh qat_deploy.sh 1 1 ToolChain

UP=/path to up
MQB=/path to mqbench

cfg=$UP/configs/quant/det/faster_rcnn/faster_rcnn_r50_fpn_improve_quant_trt_qat_deploy.yaml

jobname=qat_deploy


export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH


srun -N$1 --gres=gpu:$2 -p $3 --job-name=$jobname --cpus-per-task=2 \
nohup python -u -m up quant_deploy \
  --config=$cfg \
  > qat_deploy.txt 2>&1 &