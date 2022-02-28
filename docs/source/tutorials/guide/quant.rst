量化
====

UP基于MQBench提供quantization aware training(QAT)接口

训练超参数
----------

* batch size = the batch size of the fp pretrained model
* learning rate per image = the final convergent learning rate of the float pretrained model(YOLOX may need smaller learning rate)
* weight decay = 0
* optimizer: SGD
* epochs = 6
* lr_scheduler: MultiStepLR
* milestones: [2, 4]
* type of bn: freeze_bn
* type of head: You can train a float pretrained model with shared head while fine-tune a quantized model with an unshared head.
* training transformer: [*train_resize, *to_tensor]
* ema: False
* the template of retinanet’s qat config is configs/det/retinanet/retinanet-r18-improve_quant_trt.yaml
* the config of retinanet’s pretrained fp model is configs/det/retinanet/retinanet-r18-improve.yaml
* the template of yolox’s qat config is configs/det/retinanet/yolox_s_ret_a1_comloc_quant_trt.yaml
* the config of yolox’s pretrained fp model is configs/det/retinanet/yolox_s_ret_a1_comloc.yaml

训练
----

Quantization-aware方式基于float类型预训练模型进行训练

  .. code-block:: bash

    git clone https://github.com/ModelTC/UP.git
    git clone https://github.com/ModelTC/MQBench.git
    cd scripts
    # Follow the prompts to set config in train_quant.sh.
    sh train_quant.sh

部署
----

将量化模型转换为onnx格式

  .. code-block:: bash

    cd scripts
    # Follow the prompts to set config in quant_deploy.sh.
    sh quant_deploy.sh

benchmark
---------

参见 `quant benchmark <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/benchmark/quant_benchmark.md>`_
