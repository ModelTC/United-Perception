Quant
=====

UP supports the whole pipline of training and interfering;

`Codes <https://github.com/ModelTC/EOD/-/tree/master/up/tasks/quant>`_

Configs
-------

It contains the illustration of common configs.

`Repos <https://github.com/ModelTC/EOD/-/tree/master/configs/quant>`_

Dataset related modules
-----------------------

1. Dataset types:

  * coco

2. The type of datasets can be chosen by setting 'type' in Dataset (default is coco). The config is as followed.

  .. code-block:: yaml
    
    dataset:
      type: coco
      kwargs:
        ...

Quant setting
-------------

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


