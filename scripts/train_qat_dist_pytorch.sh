#!/bin/bash

# cmd example: sh train_qat_dist_pytorch.sh 2

UP=/path to up
MQB=/path to mqbench

cfg=$UP/configs/quant/det/retinanet/retinanet-r18-improve_quant_trt_qat.yaml


export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH


nohup python -m up train \
              --ng=$1 \
              --launch=pytorch \
              --config=$cfg \
              --display=10 > train_log.txt 2>&1 &