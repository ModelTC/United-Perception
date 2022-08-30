Segmentation
============

UP supports the whole pipline of training, deploying, and interfering;

`Codes <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/up/tasks/seg>`_

Configs
-------

It contains the illustration of common configs.

`Repos <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/seg>`_

Dataset related modules
-----------------------

1. Dataset types:

  * cityscapes

2. The type of datasets can be chosen by setting 'seg_type' in SegDataset (default is cityscapes). The config is as followed.

  .. code-block:: yaml

    dataset:
      type: cls
      kwargs:
        seg_type: cityscapes    # Default is cityscapes. Options: [cityscapes]
        meta_file: /mnt/lustre/share/HiLight/dataset/cityscapes/fine_train.txt
        image_reader:
           type: fs_opencv
           kwargs:
             image_dir: /mnt/lustre/share/HiLight/dataset/cityscapes
             color_mode: RGB
        seg_label_reader:
           type: fs_opencv
           kwargs:
             image_dir: /mnt/lustre/share/HiLight/dataset/cityscapes
             color_mode: GRAY
        transformer: [*seg_rand_resize, *flip, *seg_crop_train, *to_tensor, *normalize]
        num_classes: *num_classes
        ignore_label: 255

Deploying model
---------------

Kestrel config needs to be set while converting models:

  .. code-block:: yaml

    to_kestrel:
      toks_type: seg
      plugin: psyche
      model_name: model  # prefix of tar-model filename and model_name in meta.json 
      version: 1.0.0
      resize_hw: 640x1024
