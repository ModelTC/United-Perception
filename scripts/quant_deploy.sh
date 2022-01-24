#!/bin/bash

UP=/path to up
MQB=/path to mqbench

cfg=$UP/configs/det/retinanet/retinanet-r18-improve_quant_trt.yaml
ckpt=/path to the pytorch model trained with quant runner
input_shape="1 3 800 1216" # N C H W

jobname=deploy


export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH

g=$(($1<8?$1:8))
spring.submit run -n$1 --ntasks-per-node=$g \
                      -p $2 \
                      --job-name=$jobname \
                      --gpu \
                      --cpus-per-task=4 \
"nohup python -u -m up quant_deploy \
  --config=$cfg \
  --ckpt=$ckpt \
  --input_shape='$input_shape' \
  > quant_deploy.txt 2>&1 &"