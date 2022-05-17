Classification
==============

UP supports the whole pipline of training, deploying, and interfering;

`Codes <https://github.com/ModelTC/EOD/-/tree/master/up/tasks/cls>`_

Configs
-------

It contains the illustration of common configs and deploying.

`Repos <https://github.com/ModelTC/EOD/-/tree/master/configs/cls>`_

Dataset related modules
-----------------------

1. Dataset types:

  * imagenet
  * custom_cls

2. The type of datasets can be chosen by setting 'meta_type' in ClsDataset (default is imagenet). The config is as followed.

  .. code-block:: yaml

    dataset:
      type: cls
      kwargs:
        meta_type: imagenet    # Default is imagenet. Options: [imagenet, custom_cls]
        meta_file: train.txt
        image_reader:
           type: fs_pillow
           kwargs:
             image_dir: train
             color_mode: RGB
        transformer: [*random_resized_crop, *random_horizontal_flip, *pil_color_jitter, *to_tensor, *normalize]


Data preprocessing
------------------

UP supports additional data augmentations for classification including: Mixup，RandomErase，Mixup+Cutmix, and so on. The detail is as followed.

UP directly introduces augmentations to configs.

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


Deploying models
----------------

'CLSToKestrel' is needed when models are transformed to kestrel models as followed.

  .. code-block:: yaml

    to_kestrel:
      toks_type: cls   # settinf toks_type
      model_name: Res50
      add_softmax: False
      pixel_means: [123.675, 116.28, 103.53]
      pixel_stds: [58.395, 57.12, 57.375]
      is_rgb: True
      save_all_label: True
      type: 'UNKNOWN'


High precision baseline
-----------------------

UP supports two kinds of settings of high precision baseline of resnet including bag of tricks and resnet strikes.

Bag of tricks
^^^^^^^^^^^^^

UP imports the precision improvement way from `Bag of Tricks for Convolutional Neural Networks <https://arxiv.org/abs/1812.01187>`_ to resnet18 and resnet50. Specifically, 200 epochs, 5 epoch warmup, coslr learning rate decay, and mixup data augmentation. The mixup way is mentioned as above and the 'coslr' learning rate decay is shown as followed.
  
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

Resnet strikes
^^^^^^^^^^^^^^

UP imports the precision improvement way from `ResNet strikes back: An improved training procedure in timm <https://arxiv.org/abs/2110.00476>`_ to resnet18 and resnet50. Specifically, Random Augment increasing, 'cutmix', 'mixup', 'LAMB' optimization, 'coslr' learning rate decay, and BCE classification loss function. UP supports training configs of 100 epochs and 300 epochs. The using of 'LAMB' is as followed.


  .. code-block:: yaml

    optimizer:                 
      momentum=0.9,weight_decay=0.0001)
      type: LAMB
      kwargs:
        lr: 0.008
        weight_decay: 0.02


The using of BCE classification loss function is as followed.


  .. code-block:: yaml

    - name: post_process
    type: base_cls_postprocess
    kwargs:
       cls_loss:
         type: bce
         kwargs:
           {}


The using of Rand Augument Increasing is as followed (the augmentation can be enhanced by increasing n, m, and std.)


  .. code-block:: yaml
    
    random_augmentation: &random_augmentation
      type: torch_random_augmentationIncre
      kwargs:
        n: 2  # Randomly choosing number.
        m: 7  # The strength of each operation and the highest is 10.
        magnitude_std: 0.5  # STD of strengthes.

Knowledge distill
^^^^^^^^^^^^^^^^^

UP gets high precision resnet18 model (top1:73.04) by knowledge distilling. The teacher model is resnet152 with bag of tricks and the student model resnet18 loads the pretrained result from imagenet-1k. The config of the teacher model is as followed.


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

The config of distillation is as followed.


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

Runner chooses KDRunner during training. The config is as followed.


    .. code-block:: yaml 

    
      runtime:
        runner:
          type: kd


Illustration of downstream
--------------------------

UP supports the illustration of codes for the downstream classification task. The task needs downstream training datasets and pretrained models. The using of the dataset is as followed.


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

The config of loading pretrained models is as followed.


    .. code-block:: yaml

      saver: # Required.
        save_dir: res50_car/checkpoints/cls_std     # dir to save checkpoints
        results_dir: res50_car/results_dir/cls_std  # dir to save detection results. i.e., bboxes, masks, keypoints
        auto_resume: True  # find last checkpoint from save_dir and resume from it automatically
        pretrain_model: united-perception/res50/ckpt_latest.pth


The setting of downstream classification tasks: initial learing rate is 0.1/0.01 times of the pretrained learning rate, 150 training epochs, and 0.1 learning rate decay every 50 epochs. Specifically,


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
