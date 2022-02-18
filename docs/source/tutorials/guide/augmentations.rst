数据预处理
==========

UP 支持多种数据增广和前处理，常用数据增广包括Flip，Resize等；
前处理包括Normalization，To_Tenser，Pad。
细节如下所示：

UP 在配置文件中直接引入增广：

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

RandAug: 随机从给定的增强序列中抽取n个增强操作，并根据给定的m值和方差std来决定增强的强度

  .. code-block:: yaml
    
    random_augmentation: &random_augmentation
      type: torch_random_augmentation
      kwargs:
        n: 2  # 随机抽取的增强个数
        m: 7  # 每个增强操作的强度，最高为10
        magnitude_std: 0.5  # 强度的方差

RandAug Increasing: 原始的RandAug中有些操作，m值越小，增强的程度越大，Inceasing版本则是将所有的增强操作统一为m越大增强的强度越大

  .. code-block:: yaml
    
    random_augmentation: &random_augmentation
      type: torch_random_augmentationIncre
      kwargs:
        n: 2  # 随机抽取的增强个数
        m: 7  # 每个增强操作的强度，最高为10
        magnitude_std: 0.5  # 强度的方差

BatchPad: 经常被直接加入到 dataloader 的配置文件中。

  .. code-block:: yaml
    
    dataloader:
      type: base
      kwargs:
        num_workers: 4
        alignment: 32
        pad_value: 0
        pad_type: batch_pad

* 所有的增广都需要被加入 dataset.kwargs.transformer，如下所示：

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
