#!/bin/bash

# cmd example: sh train_ptq.sh 16 ToolChain

UP=/path to up
MQB=/path to mqbench # github mqbench commit id after 6c222c40d1a176df78bcbf4d334698185f7cd8d8

cfg=$UP/configs/quant/det/faster_rcnn/faster_rcnn_r18_FPN_2x_quant_qdrop.yaml

jobname=quant_ptq

export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH


g=$(($1<8?$1:8))
spring.submit run -n$1 --ntasks-per-node=$g \
                      -p $2 \
                      --job-name=$jobname \
                      --gpu \
                      --cpus-per-task=2 \
                      --quotatype=auto \
"nohup python -u -m up train \
  --config=$cfg \
  --display=10 \
  > train_log_ptq.txt 2>&1 &"