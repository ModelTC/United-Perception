自监督
======

UP支持自监督pretrain模型训练以及finetune下游任务;

`具体代码 <https://github.com/ModelTC/EOD/tree/main/up/tasks/ssl>`_

配置文件
--------

* `pretrain <https://github.com/ModelTC/EOD/blob/main/configs/ssl/mocov1/moco_v1.yaml>`_
* `finetune <https://github.com/ModelTC/EOD/blob/main/configs/ssl/mocov1/moco_v1_imagenet_linear.yaml>`_

pretrain相关模块
----------------

以下为pertain配置文件示例:

  .. code-block:: yaml

    # pretrain骨干网络配置
    net: &subnet
      - name: backbone
        type: ssl  # 自监督自任务
        multi_model:  # 支持多个模型的骨干网络结构
          - name: encoder_q
            type: resnet50  # 骨干网络类型（cls模块中的模型）
            kwargs:  # 其他参数
              frozen_layers: []
              ...
          - name: encoder_k
            type: resnet50  # 骨干网络类型（cls模块中的模型）
            kwargs:  # 其他参数
              frozen_layers: []
              ...
        wrappers:
          - type: moco  # 自监督模型，支持moco, simclr, simsiam
            kwargs:
              dim: 128
              K: 65536

      - name: post_process
        type: base_ssl_postprocess  # 自监督后处理模块
        kwargs:
           ssl_loss:  # 自监督损失函数
             type: moco_loss

finetune相关模块
----------------

以下为finetune配置文件示例:

    .. code-block:: yaml

        dataset:
          type: cls
          kwargs:
            meta_file: ...
            image_reader:
               ...
            transformer: ...
            fraction: 0.1  # 做finetune的标签比例

        saver:
          save_dir: moco_v1_linear_dist8/checkpoints/cls_std
          results_dir: moco_v1_linear_dist8/results_dir/cls_std
          auto_resume: True
          pretrain_model: cls_std/ckpt.pth  # pretrain模型参数地址

