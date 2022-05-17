量化
====

UP支持量化任务训练、推理的全部流程;
`具体代码 <https://github.com/ModelTC/EOD/-/tree/master/up/tasks/quant>`_

配置文件
--------

`代码仓库 <https://github.com/ModelTC/EOD/-/tree/master/configs/quant>`_
其中包括常用算法配置文件

数据集相关模块
--------------

1. 数据集类型包括:

  * coco

2. 数据集类型通过设置Dataset的type来选择，默认为coco，配置文件示例如下:

  .. code-block:: yaml
    
    dataset:
      type: coco
      kwargs:
        ...

量化设置
--------

  .. code-block:: yaml

    quant:
      ptq_only: False
      deploy_backend: tensorrt
      cali_batch_size: 900
      prepare_args:
        extra_qconfig_dict:
          w_observer: MinMaxObserver
          a_observer: EMAMinMaxObserver
          w_fakequantize: FixedFakeQuantize
          a_fakequantize: FixedFakeQuantize
        leaf_module: [Space2Depth, FrozenBatchNorm2d]
        extra_quantizer_dict:
          additional_module_type: [ConvFreezebn2d, ConvFreezebnReLU2d]


