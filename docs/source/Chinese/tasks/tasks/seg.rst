分割
====

UP支持分割任务训练、部署、推理的全部流程;
`具体代码 <https://github.com/ModelTC/EOD/-/tree/master/up/tasks/seg>`_

配置文件
--------

`代码仓库 <https://github.com/ModelTC/EOD/-/tree/master/configs/seg>`_
其中包括常用算法配置文件

数据集相关模块
--------------

1. 数据集类型包括:

  * cityscapes

2. 数据集类型通过设置SegDataset的seg_type来选择，默认为cityscapes，配置文件示例如下:

  .. code-block:: yaml

    dataset:
      type: cls
      kwargs:
        seg_type: cityscapes    # 默认为cityscapes，选项包括: [cityscapes]
        meta_file: cityscapes/fine_train.txt
        image_reader:
           type: fs_opencv
           kwargs:
             image_dir: cityscapes
             color_mode: RGB
        seg_label_reader:
           type: fs_opencv
           kwargs:
             image_dir: cityscapes
             color_mode: GRAY
        transformer: [*seg_rand_resize, *flip, *seg_crop_train, *to_tensor, *normalize]
        num_classes: *num_classes
        ignore_label: 255

部署模块
--------

转换kestrel模型时，需要设置具体配置如下:

  .. code-block:: yaml

    to_kestrel:
      toks_type: seg
      plugin: psyche
      model_name: model  # tar模型文件名的前缀以及meta.json中的model_name
      version: 1.0.0
      resize_hw: 640x1024
