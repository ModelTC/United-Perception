检测
====

UP支持检测任务训练、部署、推理的全部流程;
`具体代码 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/up/tasks/det>`_

配置文件
--------

`代码仓库 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/det>`_
其中包括常用算法配置文件与部署示例

数据集相关模块
--------------

1. 数据集类型包括:

  * coco
  * custom
  * lvis

2. 数据集类型，配置如下:

  .. code-block:: yaml

    dataset:
      type: coco   # [coco, custom, lvis]
      kwargs:
        meta_file: /mnt/lustre/share/DSK/datasets/mscoco2017/annotations/instances_train2017.json
        image_reader:
          type: fs_opencv
          kwargs:
            image_dir: /mnt/lustre/share/DSK/datasets/mscoco2017/train2017
            color_mode: RGB
        transformer: [*stitch_expand, *crop, *flip, *color_jitter, *fix_output_resize,*to_tensor, *normalize]

3. 数据增广:

  * StitchExpand:

  .. code-block:: yaml

    expand: &stitch_expand
      type: stitch_expand
      kwargs:
        expand_ratios: 2.0
        expand_prob: 0.5

  * ImageCrop:

  .. code-block:: yaml

    crop: &crop
      type: crop
      kwargs:
        means: [123.675, 116.280, 103.530]
        scale: 1024
        crop_prob: 0.5

损失函数
--------

UP支持focal loss，iou loss 和 smooth l1 loss

focal loss:

  * sigmoid_focal_loss

  .. code-block:: yaml
  
    loss:
      type: sigmoid_focal_loss
      kwargs:
        num_classes: *num_classes
        alpha: 0.25
        gamma: 2.0
        init_prior: 0.01

  * quality_focal_loss

  .. code-block:: yaml

    loss:
      type: quality_focal_loss
      kwargs:
        gamma: 2.0
        init_prior: 0.01

iou loss:

  .. code-block::

    loss:
      type: iou_loss
      kwargs:
        loss_type: iou  # type: ['iou', 'giou', 'diou', 'ciou', 'linear_iou', 'square_iou']

smooth_l1_loss:

  .. code-block::

    loss:
      type: smooth_l1_loss
      kwargs:
        sigma: 3.0

compose_loc_loss:

  .. code-block::

    loss:
      type: compose_loc_loss
      kwargs:
        loss_cfg:
          - type: iou_loss
            kwargs:
              loss_type: giou
              loss_weight: 1.0
          - type: l1_loss
            kwargs:
              loss_weight: 1.0

部署模块
--------

转换kestrel模型时，需要使用ToKestrel，具体配置如下:

  .. code-block:: yaml

    to_kestrel:
      detector: FASTER-RCNN  # choices [RETINANET, RFCN, FASTER-RCNN]
      save_to: kestrel_model  # saved file
      default_confidence_thresh: 0.3
      plugin: harpy   # choices = [essos, harpy, sphinx]
      version: 1.0.0
      resize_hw: 640x1024
      kestrel_config:   # 可选
        -
          # harpy
          confidence_thresh: 0.3
          # essos
          thresh: 0.3
          id: 37017
          label: face
          filter_h: 0
          filter_w: 0