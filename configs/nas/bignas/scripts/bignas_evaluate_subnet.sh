#!/bin/bash

UP=PATH_TO_UP

cfg=configs/nas/bignas/det/bignas_retinanet_R18_evaluate_subnet.yaml

jobname=bignas_evaluate_subnet

export PYTHONPATH=$UP:$PYTHONPATH


g=$(($1<8?$1:8))
spring.submit run -n$1 --ntasks-per-node=$g \
                      -p $2 \
                      --job-name=$jobname \
                      --gpu \
                      --cpus-per-task=2 \
                      --quotatype=auto \
"python -u -m up train \
  --phase=evaluate_subnet \
  --config=$cfg \
  --display=10 \
  2>&1 | tee eval_subnet_log_bignas.txt"
