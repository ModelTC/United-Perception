分类
====

UP支持分类任务训练、部署、推理的全部流程;
`具体代码 <https://github.com/ModelTC/United-Perception/tree/main/up/tasks/cls>`_

配置文件
--------

`代码仓库 <https://github.com/ModelTC/United-Perception/tree/main/configs/cls>`_
其中包括常用算法配置文件与部署示例

数据集相关模块
--------------

1. 数据集类型包括:

  * imagenet
  * custom_cls

2. 数据集类型通过设置ClsDataset的meta_type来选择，默认为imagenet，配置文件示例如下:

  .. code-block:: yaml

    dataset:
      type: cls
      kwargs:
        meta_type: imagenet    # 默认为imagenet，选项包括: [imagenet, custom_cls]
        meta_file: train.txt
        image_reader:
           type: fs_pillow
           kwargs:
             image_dir: train
             color_mode: RGB
        transformer: [*random_resized_crop, *random_horizontal_flip, *pil_color_jitter, *to_tensor, *normalize]


数据预处理
----------

UP 的分类模块提供了额外的数据增广函数，数据增广包括：Mixup，RandomErase，Mixup+Cutmix等；
细节如下所示：

UP 在配置文件中直接引入增广：

Mixup:

  .. code-block:: yaml

    mixup: &mixup
      type: torch_mixup
      kwargs:
        alpha: 0.2
        num_classes: 1000
        extra_input: True


RandomErase:

  .. code-block:: yaml

    rand_erase: &rand_erase
      type: torch_randerase
      kwargs:
        probability: 0.25


Cutmix + Mixup:

  .. code-block:: yaml

    cutmix_mixup: &cutmix_mixup
      type: torch_cutmix_mixup
      kwargs:
        mixup_alpha: 0.1
        cutmix_alpha: 1.0
        switch_prob: 0.5
        num_classes: 1000
        extra_input: True
        transform: True


部署模块
--------

转换kestrel模型时，需要使用CLSToKestrel，具体配置如下:

  .. code-block:: yaml

    to_kestrel:
      toks_type: cls   # 通过设置toks_type
      plugin: classifier
      model_name: model  # tar模型文件名的前缀以及meta.json中的model_name
      pixel_means: [123.675, 116.28, 103.53]
      pixel_stds: [58.395, 57.12, 57.375]
      is_rgb: True
      save_all_label: True
      type: 'UNKNOWN


高精度baseline
--------------

UP提供了两种resnet的高精度baseline设置，分别是bag of tricks和resnet strikes

bag of tricks
^^^^^^^^^^^^^
UP将论文 `Bag of Tricks for Convolutional Neural Networks <https://arxiv.org/abs/1812.01187>`_ 中提到的提点技巧引入到了resnet18和resnet50中，分别为：200epoch的训练时长，5个epoch的warmup，coslr学习率余弦衰减以及mixup数据增强方法。其中mixup数据增强方法如上文所示，coslr学习率余弦衰减具体配置如下：
  
  .. code-block:: yaml

    lr_scheduler:
      warmup_iter: 3130
      warmup_type: linear
      warmup_register_type: no_scale_lr
      warmup_ratio: 0.25
      type: CosineAnnealingLR
      kwargs:
          T_max: 200
          eta_min: 0.0
          warmup_iter: 3130

resnet strikes
^^^^^^^^^^^^^^
UP将论文 `ResNet strikes back: An improved training procedure in timm <https://arxiv.org/abs/2110.00476>`_ 中的提点技巧引入到了resnet18和resnet50中，分别为：Random Augment Increasing, cutmix, mixup, LAMB优化器，coslr学习率余弦衰减以及BCE分类损失，并分别提供了100epoch和300epoch两种训练时长下的配置文件。其中，LAMB的使用配置如下：


  .. code-block:: yaml

    optimizer:                 
      momentum=0.9,weight_decay=0.0001)
      type: LAMB
      kwargs:
        lr: 0.008
        weight_decay: 0.02


BCE分类损失的使用配置如下：


  .. code-block:: yaml

    - name: post_process
    type: base_cls_postprocess
    kwargs:
       cls_loss:
         type: bce
         kwargs:
           {}


Rand Augument Increasing的使用配置如下（可通过增大n，m和std来增加增强的强度）：


  .. code-block:: yaml
    
    random_augmentation: &random_augmentation
      type: torch_random_augmentationIncre
      kwargs:
        n: 2  # 随机抽取的增强个数
        m: 7  # 每个增强操作的强度，最高为10
        magnitude_std: 0.5  # 强度的方差

knowledge distill
^^^^^^^^^^^^^^^^^
UP通过知识蒸馏，获得到resnet18的高精度模型（top1:73.04)。选用 resnet152 with bag of tricks作为教师模型，学生模型resnet18同时也加载imagenet-1k预训练结果。教师模型的配置文件为：


    .. code-block:: yaml

      teacher: 
        - name: backbone              # backbone = resnet50(frozen_layers, out_layers, out_strides)
          type: resnet152
          kwargs:
            frozen_layers: []
            out_layers: [4]     # layer1...4, commonly named Conv2...5
            out_strides: [32]  # tell the strides of output features
            normalize:
              type: solo_bn
            initializer:
              method: msra
            deep_stem: True 
            avg_down: True
        - name: head
          type: base_cls_head
          kwargs:
            num_classes: *num_classes
            in_plane: &teacher_out_channel 2048
            input_feature_idx: -1

蒸馏的任务定义配置文件为：


    .. code-block:: yaml

      mimic:
        mimic_name: res152_to_res18
        mimic_type: kl
        loss_weight: 1.0 
        teacher:
          mimic_name: ['head.classifier']
          teacher_weight: /UP/resnet152_tricks/teacher.pth.tar
        student:
          mimic_name: ['head.classifier']
          student_weight: /UP/res18_s/res18.pth.tar

训练时，runner选择KDRunner，配置使用为：


    .. code-block:: yaml 

    
      runtime:
        runner:
          type: kd


downstream 示例
----------------
UP提供了resnet50在下游分类任务的示例代码，训练下游任务一般要准备下游训练数据集，加载预训练模型。数据集的使用配置为：


  .. code-block:: yaml


    dataset:
      type: cls
      kwargs:
        meta_type: custom_cls
        meta_file: /cars_im_folder//train.txt
        image_reader:
           type: fs_pillow
           kwargs:
             image_dir: /cars_im_folder/train
             color_mode: RGB
        transformer: [*random_resized_crop, *random_horizontal_flip, *pil_color_jitter, *to_tensor, *normalize]

预训练模型加载配置为：


    .. code-block:: yaml

      saver: # Required.
        save_dir: res50_car/checkpoints/cls_std     # dir to save checkpoints
        results_dir: res50_car/results_dir/cls_std  # dir to save detection results. i.e., bboxes, masks, keypoints
        auto_resume: True  # find last checkpoint from save_dir and resume from it automatically
        pretrain_model: united-perception/res50/ckpt_latest.pth


UP提供的基准下游分类任务的配置为：初始学习率为预训练学习率的0.1/0.01，训练150epoch，每50epoch衰减一次学习率（lr * 0.1），具体的配置为：


    .. code-block:: yaml


        optimizer:                
          type: SGD
          kwargs:
            lr: 0.01
            nesterov: True
            momentum: 0.9
            weight_decay: 0.0005
        lr_scheduler:              
          warmup_iter: 0          # 1000 iterations of warmup
          warmup_type: linear
          warmup_register_type: no_scale_lr
          warmup_ratio: 0.25
          type: MultiStepLR
          kwargs:
            milestones: [50, 100]     # [60000, 80000]
            gamma: 0.1      
