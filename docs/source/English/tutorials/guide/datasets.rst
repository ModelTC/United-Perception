Data configuration and customization
====================================

UP supports two types of datasets: the public dataset and the custom dataset.

Public dataset
--------------

Currently, the following public datasets are supported.

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

* You need to set the 'meta_file' and 'image_dir' in dataset, and set augmentation in transformer.
* UP seperates the dataset and evalutor for adapting different kinds of datasets and evalutors.
* UP supports two kinds of evalutors for different datasets: coco(CocoEvaluator) - CocoDataset, MREvaluator(CustomDat    aset) - CustomDataset.

Training on the custom dataset
------------------------------


Dataset structure
-----------------

The structure of dataset is as followed.

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

Annotation file format
----------------------

We suggest that saving the annotation file with json format. 
A single image is annotated as followed.

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

Accuracy evalution
------------------

Config is as followed.

  .. code-block:: yaml

    evaluator:
      type: MR
      kwargs:
        gt_file: path/your/test.json
        iou_thresh: 0.5
        num_classes: *num_classes
        data_respective: False

  .. note::

    * UP supports multiple training/testing evalutions. When 'data_respective' is set to True,the datasets will be respectively evaluted; when it is set to False, the datasets will be together evaluted.
    * When 'data_respective' is set to True, we suggest setting the batchsize to 1, or the padding between different data will effect the final accuracy.

UP supports evaluting the custom dataset by MR mode which contains two indices as followed.

  * MR@FPPI=xxx: Miss rate while 'FPPI' reaches some value.
  * Score@FPPI=xxx: Confidence score while 'FPPI' reaches some value.

Illustration of the config
--------------------------

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

    * The types of the dataset and evalution need to be set to **custom** and **MR**
    * The parameter 'num_classes' needs to be set.
