#!/bin/bash

# cmd example: sh quant_deploy.sh 1 ToolChain

UP=/path to up
MQB=/path to mqbench

cfg=$UP/configs/det/faster_rcnn/faster_rcnn_r50_fpn_improve_trt_quant_deploy.yaml

jobname=deploy


export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH

g=$(($1<8?$1:8))
spring.submit run -n$1 --ntasks-per-node=$g \
                      -p $2 \
                      --job-name=$jobname \
                      --gpu \
                      --cpus-per-task=2 \
"nohup python -u -m up quant_deploy \
  --config=$cfg \
  > quant_deploy.txt 2>&1 &"