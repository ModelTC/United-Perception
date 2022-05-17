Detection
=========

UP supports the whole pipline of training, deploying, and interfering;

`Codes <https://github.com/ModelTC/EOD/-/tree/master/up/tasks/det>`_

Configs
-------

It contains the illustration of common configs and deploying.

`Repos <https://github.com/ModelTC/EOD/-/tree/master/configs/det>`_

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
        meta_file: coco/annotations/instances_train2017.json
        image_reader:
          type: fs_opencv
          kwargs:
            image_dir: coco/train2017
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

  .. code-block:: yaml

    loss:
      type: iou_loss
      kwargs:
        loss_type: iou  # type: ['iou', 'giou', 'diou', 'ciou', 'linear_iou', 'square_iou']

smooth_l1_loss:

  .. code-block:: yaml

    loss:
      type: smooth_l1_loss
      kwargs:
        sigma: 3.0

compose_loc_loss:

  .. code-block:: yaml

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

1. Add 'class_names' to dataset.train, take COCO as an example:

  .. code-block:: yaml

    class_names: &class_names [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
    ]

    dataset:
      train:
        dataset:
          type: coco
          kwargs:
            meta_file: coco/annotations/instances_train2017.json
            class_names: *class_names
            image_reader:
              type: fs_opencv
              kwargs:
                image_dir: coco/train2017
                color_mode: RGB
            transformer: [*flip, *train_resize, *to_tensor, *normalize]

2. Add 'to_kestrel' to config:

  .. code-block:: yaml

    to_kestrel:
      detector: FASTER-RCNN  # choices [RETINANET, RFCN, FASTER-RCNN]
      model_name: model  # prefix of tar-model filename and model_name in meta.json
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
