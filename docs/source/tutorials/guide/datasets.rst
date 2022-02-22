数据相关配置与自定义
====================

UP 支持两种格式的数据集：公用数据集（public datasets）和自定义数据集（custom dataset）。

公用数据集
----------

目前，下列形式的公开数据集是被支持的：

CocoDataset

  .. code-block:: yaml

    dataset: # Required.
      train:
        dataset:
          type: coco
          kwargs:
            meta_file: coco/annotations/instances_train2017.json
            image_reader:
              type: fs_opencv   # ['fs_opencv', 'fs_pillow', 'ceph_opencv', 'osg']
              kwargs:
                image_dir: coco/train2017
                color_mode: RGB
            transformer: [*flip, *train_resize, *to_tensor, *normalize]
      test:
        dataset:
          type: coco
          kwargs:
            meta_file: &gt_file coco/annotations/instances_val2017.json
            image_reader:
              type: fs_opencv   # ['fs_opencv', 'fs_pillow', 'ceph_opencv', 'osg']
              kwargs:
                image_dir: coco/val2017
                color_mode: RGB
            transformer: [*test_resize, *to_tensor, *normalize]
            evaluator:
              type: COCO               # choices = {'COCO'}
              kwargs:
                gt_file: *gt_file
                iou_types: [bbox]
      batch_sampler:
        type: aspect_ratio_group
        kwargs:
          sampler:
            type: dist
            kwargs: {}
          batch_size: 2
          aspect_grouping: [1,]
      dataloader:
        type: base
        kwargs:
          num_workers: 4
          alignment: 32

* 你需要设置元文件（meta_file）和图像目录（image_dir）到数据集中，设置数据增广（augmentations）到变换器（transformer）中。
* UP 将数据集和评估器（evaluator）分开以适应多种不同的数据集和评估器.
* UP 对不同的数据集支持两种评估器：coco(CocoEvaluator) - CocoDataset, MREvaluator(CustomDataset) - CustomDataset.

在自定义数据集上训练
--------------------

数据集结构
----------

数据集结构组成示例如下:

  .. code-block:: bash

    datasets 
    ├── image_dir 
    |   ├── train 
    |   ├── test 
    |   └── val 
    └── annotations     
        ├── train.json     
        ├── test.json     
        └── val.json

标注文件格式
------------

建议以json格式保存标注文件，单张图像标注格式如下:

  .. code-block:: yaml

    {
        "filename": "000005.jpg",
        "image_height": 375,
        "image_width": 500,
        "instances": [                // List of labeled entities, optional for test
          {
            "is_ignored": false,
            "bbox": [262,210,323,338], // x1,y1,x2,y2
            "label": 9                 // Label id start from 1, total C classes corresponding  a range of [1, 2, ..., C]
          },
          {
            "is_ignored": false,
            "bbox": [164,263,252,371],
            "label": 9
          },
          {
            "is_ignored": false,
            "bbox": [4,243,66,373],
            "label": 9
          }    
        ]
    }

精度评估
--------

配置文件示例:

  .. code-block:: yaml

    evaluator:
      type: MR
      kwargs:
        gt_file: path/your/test.json
        iou_thresh: 0.5
        num_classes: *num_classes
        data_respective: False

  .. note::

    * UP支持多验证\测试集评估，通过设置data_respective为True，支持多数据集分别评估精度，为False时，多数据集统一评估精度
    * 当data_respective被设置为True时，建议将测试集的batch size设置为1，否则会由于不同数据间的padding影响最终精度

UP支持MR模式评估自定义数据集精度，包括两个指标:

  * MR@FPPI=xxx: Miss rate while FPPI reaches some value.
  * Score@FPPI=xxx: Confidence score while FPPI reaches some value.

配置文件示例
------------

  .. code-block:: yaml

    dataset:
      train:
        dataset:
          type: custom
          kwargs:
            num_classes: *num_classes
            meta_file: # fill in your own train annotation file
            image_reader:
              type: fs_opencv
              kwargs:
                image_dir: # fill in your own train data path
                color_mode: RGB
            transformer: [*flip, *resize, *to_tensor, *normalize]
      test:
        dataset:
          type: custom
          kwargs:
            num_classes: *num_classes
            meta_file: # fill in your own test annotation file 
            image_reader:
              type: fs_opencv
              kwargs:
                image_dir: # fill in your own test data path  
                color_mode: RGB
            transformer: [*resize, *to_tensor, *normalize]
            evaluator:
              type: MR # fill in your own evaluator
              kwargs:
                gt_file: # fill in your own test annotation file
                iou_thresh: 0.5
                num_classes: *num_classes
      batch_sampler:
        type: aspect_ratio_group
        kwargs:
          sampler:
            type: dist
            kwargs: {}
          batch_size: 2
          aspect_grouping: [1,]
      dataloader:
        type: base
        kwargs:
          num_workers: 4
          alignment: 1

  .. note::

    * 数据集与评估方法类型需要设定为 **custom** and **MR**
    * 需要设置num_classes参数
