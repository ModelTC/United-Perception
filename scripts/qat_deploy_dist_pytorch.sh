#!/bin/bash

# cmd example: sh qat_deploy_dist_pytorch.sh

UP=/path to up
MQB=/path to mqbench

cfg=$UP/configs/quant/det/faster_rcnn/faster_rcnn_r50_fpn_improve_quant_trt_qat_deploy.yaml


export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH

nohup python -m up quant_deploy \
              --config=$cfg  > deploy_log.txt 2>&1 &