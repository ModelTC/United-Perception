Training on Custom Data
=======================

Custom dataset
--------------

.. code-block:: yaml
  
  num_classes: &num_classes 15

  flip: &flip
    type: flip
    kwargs:
      flip_p: 0.5

  to_tensor: &to_tensor
    type: custom_to_tensor

  train_resize: &train_resize
    type: keep_ar_resize_max
    kwargs:
      padding_type: left_top
      padding_val: 0
      random_size: [10, 20]

  test_resize: &test_resize
    type: keep_ar_resize_max
    kwargs:
      max_size: 416
      padding_type: left_top
      padding_val: 0


  dataset:
    train:
      dataset:
        type: custom
        kwargs:
          num_classes: *num_classes
          meta_file: path/your/train.json
          image_reader:
            type: fs_opencv
            kwargs:
              image_dir: &image_dir path/your/train_image_dir
              color_mode: BGR
          transformer: [*flip, *train_resize, *to_tensor]
      batch_sampler:
        type: base
        kwargs:
          sampler:
            type: dist
            kwargs:
              fix_seed: False
          batch_size: 8
    test:
      dataset:
        type: custom
        kwargs:
          num_classes: *num_classes
          meta_file: &gt_file path/your/test.json
          image_reader:
            type: fs_opencv
            kwargs:
              image_dir: path/your/test_image_dir
              color_mode: BGR
          transformer: [*test_resize, *to_tensor]
          evaluator:
            type: MR
            kwargs:
              gt_file: *gt_file
              iou_thresh: 0.5
              num_classes: *num_classes
      batch_sampler:
        type: base
        kwargs:
          sampler:
            type: dist
            kwargs: {}
          batch_size: 8
    dataloader:
      type: base
      kwargs:
        num_workers: 4
        alignment: 32
        worker_init: True
        pad_type: batch_pad

Rank dataset
------------

.. code-block:: yaml
  
  num_classes: &num_classes 10

  flip: &flip
    type: flip
    kwargs:
      flip_p: 0.5

  to_tensor: &to_tensor
    type: custom_to_tensor

  train_resize: &train_resize
    type: keep_ar_resize_max
    kwargs:
      padding_type: left_top
      padding_val: 0
      random_size: [10, 20]

  test_resize: &test_resize
    type: keep_ar_resize_max
    kwargs:
      max_size: 416
      padding_type: left_top
      padding_val: 0


  dataset:
    train:
      dataset:
        type: rank_custom
        kwargs:
          num_classes: *num_classes
          meta_file: path/your/train.json
          image_reader:
            type: fs_opencv
            kwargs:
              image_dir: &image_dir path/your/train_image_dir
              color_mode: BGR
          transformer: [*flip, *train_resize, *to_tensor]
      batch_sampler:
        type: base
        kwargs:
          sampler:
            type: local # special for rank_dataset
            kwargs: {}
          batch_size: 8
    test:
      dataset:
        type: custom
        kwargs:
          num_classes: *num_classes
          meta_file: &gt_file path/your/test.json
          image_reader:
            type: fs_opencv
            kwargs:
              image_dir: path/your/test_image_dir
              color_mode: BGR
          transformer: [*test_resize, *to_tensor]
          evaluator:
            type: MR
            kwargs:
              gt_file: *gt_file
              iou_thresh: 0.5
              num_classes: *num_classes
          reload_cfg:  # special for rank_dataset
            mini_epoch: 1
            seed: 0
            group: &group 1 # to avoid memory overflow, generate random numbers in groups
      batch_sampler:
        type: base
        kwargs:
          sampler:
            type: dist
            kwargs: {}
          batch_size: 8
    dataloader:
      type: base
      kwargs:
        num_workers: 4
        alignment: 32
        worker_init: True
        pad_type: batch_pad

  # special for rank_dataset
  hooks:
    - type: reload
      kwargs: 
        group: *group

Dataset Structure
-----------------

The following format is highly recommended:

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

Annotation Format
-----------------

Annotation file is stored in JSON format, and each piece of data should be organized as:

  .. code-block:: bash
   
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

Custom Evaluator
-----------------

UP evaluates custom dataset with MR mode which contains two metrics:
* MR@FPPI=xxx: Miss rate while FPPI reaches some value.
* Score@FPPI=xxx: Confidence score while FPPI reaches some value.

Configuration File
------------------

Here is an example:

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

* Dataset and evaluator type should be set as custom and MR.
* Num_classes need to be set.
* Training, evaluating and inferencing commonds refer to Get Started.

