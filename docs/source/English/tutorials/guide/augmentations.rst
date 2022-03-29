Data preprocessing
==================

UP supports various data augmentations and preprocesses.
Data augmentation contains Flip, Resize, and so on.
Preprocesses contains Normlization, To_Tensor, Pad, and so on.
The detail is as followed.

UP directly imports data augmentation in configs.

Flip:

  .. code-block:: yaml

    flip: &flip   
      type: flip
      kwargs:
        flip_p: 0.5

Resize:

  .. code-block:: yaml
    
    resize: &train_resize
      type: keep_ar_resize
      kwargs:
        scales: [640, 672, 704, 736, 768, 800]
        max_size: 1333
        separate_wh: True

Normalization:

  .. code-block:: yaml

    normalize: &normalize
      type: normalize
      kwargs:
        mean: [0.485, 0.456, 0.406] # ImageNet pretrained statics
        std: [0.229, 0.224, 0.225]

ToTensor:

  .. code-block:: yaml
    
    to_tensor: &to_tensor
      type: to_tensor

RandAug: Randomly choose n augmentation operations from the given augmentation list, and use mean (m) and variance (std) to adjust their augmentation weights.

  .. code-block:: yaml
    
    random_augmentation: &random_augmentation
      type: torch_random_augmentation
      kwargs:
        n: 2  # Randomly chosen operations
        m: 7  # the average of weights of augmentations, the highest value can be set to 10
        magnitude_std: 0.5  # the variance of weights of augmentations

RandAug Increasing: Some operations in the original 'RandAug' have the case of stronger augmentation with smaller m. The Increasing version revises the case and ensures that all operations have stronger augmentation with bigger m.

  .. code-block:: yaml
    
    random_augmentation: &random_augmentation
      type: torch_random_augmentationIncre
      kwargs:
        n: 2  # Randomly chosen operations
        m: 7  # the average of weights of augmentations, the highest value can be set to 10
        magnitude_std: 0.5  # the variance of weights of augmentations

BatchPad: It is usually directly added into the config of dataloader.

  .. code-block:: yaml
    
    dataloader:
      type: base
      kwargs:
        num_workers: 4
        alignment: 32
        pad_value: 0
        pad_type: batch_pad

All augmentations need to be written into dataset.kwargs.transformer as followed.

  .. code-block:: yaml
    
    dataset:
      type: coco
      kwargs:
        meta_file: coco/annotations/instances_train2017.json
        image_reader:
          type: fs_opencv
          kwargs:
            image_dir: coco/train2017
            color_mode: RGB
        transformer: [*flip, *train_resize, *to_tensor, *normalize]   # add here in order
