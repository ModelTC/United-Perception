#!/bin/bash

UP=/path to up
MQB=/path to mqbench

cfg=$UP/configs/quant/det/retinanet/retinanet-r18-improve_quant_trt_qat.yaml

export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH


nohup python -u -m up train \
 --config=$cfg \
 --display=10 \
 > train_log_qat.txt 2>&1 &
