We provide quantization aware training(QAT) interface based on MQBench.

## Get Started

### Training hyper-parameters for example

- batch size = the batch size of the fp pretrained model
- learning rate per image = the final convergent learning rate of the float pretrained model(YOLOX may need smaller learning rate)
- weight decay = 0
- optimizer: SGD
- epochs = 6
- lr_scheduler: MultiStepLR
- milestones: [2, 4]
- type of bn: freeze_bn
- type of head: You can train a float pretrained model with shared head while fine-tune a quantized model with an unshared head.
- training transformer: [*train_resize, *to_tensor]
- ema: False
- the template of retinanet’s qat config is configs/det/retinanet/retinanet-r18-improve_quant_trt.yaml
- the config of retinanet’s pretrained fp model is configs/det/retinanet/retinanet-r18-improve.yaml
- the template of yolox’s qat config is configs/det/retinanet/yolox_s_ret_a1_comloc_quant_trt.yaml
- the config of yolox’s pretrained fp model is configs/det/retinanet/yolox_s_ret_a1_comloc.yaml


### train
Quantization-aware train a model based on a float pretrained model.

```shell
git clone https://github.com/ModelTC/EOD.git
git clone https://github.com/ModelTC/MQBench.git
cd scripts
# Follow the prompts to set config in train_quant.sh.
sh train_quant.sh
```

### deploy
export the quantized model to ONNX

```shell
cd scripts
# Follow the prompts to set config in quant_deploy.sh.
sh quant_deploy.sh
```

### result


| Model                 | map\@fp32 | ptq\int8 | qat\int8 |
|-----------------------|-----------|----------|----------|
| retinanet-r18-improve | 40.7      | 40.5     | 40.7     |
| yolox_s               | 40.4      | 39.4     | 39.8     |
