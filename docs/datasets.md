# Dataset
EOD supports two types: public datasets and custom dataset.

## Public Datasets
Currently, the following dataset types are supported:

### CocoDataset

```yaml
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
```

* You need to set meta_file and image_dir into **dataset** and augmentations into **transformer**.
* EOD separates dataset and evaluator to adapt various datasets and evaluators
* EOD supports two evaluators for various datasets: coco(CocoEvaluator) - CocoDataset, MREvaluator(CustomDataset) - CustomDataset.

## Custom Dataset
EOD support custom dataset. Details could refer to [Training on custom data](train_custom_data.md). 
