数据相关配置与自定义/Datasets
=============================

UP 支持两种格式的数据集：公开数据集（public datasets）和定制数据集（custom dataset）。

公开数据集（Public Datasets）
----------------------------

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
              type: fs_opencv
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
              type: fs_opencv
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

在定制数据集上训练
------------------

UP 支持定制数据集。细节可以在`Training on custom data <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/master/docs/train_custom_data.md>`_中查询。

