Detection
=========

UP supports the whole pipline of training, deploying, and interfering;

`Codes <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/up/tasks/det>`_

Configs
-------

It contains the illustrtion of common configs and deploying.

`Repos <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/det>`_

Dataset related modules
-----------------------

1. Dataset types:

  * coco
  * custom
  * lvis

2. The type of datasets and the config is as followed:

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

3. Data augmentation:

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

Loss fuction
------------

UP supports focal loss, iou loss, and smooth l1 loss.

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

Deploying modules
-----------------

'ToKestrel' is needed when models are transformed to kestrel models as followed.

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