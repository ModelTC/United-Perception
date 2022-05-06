#!/bin/bash

# cmd example: sh train_qat.sh 16 ToolChain

UP=/path to up
MQB=/path to mqbench

cfg=$UP/configs/quant/det/retinanet/retinanet-r18-improve_quant_trt_qat.yaml


jobname=quant_qat


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
  > train_log_qat.txt 2>&1 &"