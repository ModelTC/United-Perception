self-supervision
====

UP supports self-supervision to train pretrained model and finetune downstream task;

`Codes <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/up/tasks/ssl>`_

Config
--------

* `pretrain <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/configs/ssl/mocov1/moco_v1.yaml>`_
* `finetune <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/configs/ssl/mocov1/moco_v1_imagenet_linear.yaml>`_

pretrain module
--------------

The following snippet shows an example config for pretrain task.

  .. code-block:: yaml

    # pretrain backbone config
    net: &subnet
      - name: backbone
        type: ssl  # self-supervision sub-task
        multi_model:  # support multiple backbones
          - name: encoder_q
            type: resnet50  # type of backbone (found in cls module)
            kwargs:  # 其他参数
              frozen_layers: []
              ...
          - name: encoder_k
            type: resnet50  # type of backbone (found in cls module)
            kwargs:  # other hyperparameters
              frozen_layers: []
              ...
        wrappers:
          - type: moco  # self-supervision model, support moco, simclr and simsiam
            kwargs:
              dim: 128
              K: 65536

      - name: post_process
        type: base_ssl_postprocess  # postprocessing module for self-supervision
        kwargs:
           ssl_loss:  # loss function for self-supervision
             type: moco_loss

finetune module
--------------

The following snippet shows an example config for finetune task:

    .. code-block:: yaml

        saver:
          save_dir: moco_v1_linear_dist8/checkpoints/cls_std
          results_dir: moco_v1_linear_dist8/results_dir/cls_std
          auto_resume: True
          pretrain_model: /mnt/lustre/baishihao/UP_MOCO/moco_v1_bs512/checkpoints/cls_std/ckpt.pth  # dir of pretrained model parameters