Keypoints detection
===================

UP supports the whole pipline of training, deploying, and interfering;

`Codes <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/up/tasks/kp>`_

Configs
-------

It contains the illustrtion of common configs

`Repos <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/kp>`_

Dataset related modules
-----------------------

1. Dataset types:

  * keypoint

Data preprocessing
------------------

UP supports additional data augmentations for keypoints detection including 'keep_aspect_ratio', 'random_scale', and so on. The detail is as followed.

keep_aspect_ratio:

  .. code-block:: yaml

    keep_aspect_ratio: &keep_aspect_ratio
      type: kp_keep_aspect_ratio
      kwargs:
        in_h: 256
        in_w: 192

random_scale:

  .. code-block:: yaml

    random_scale: &random_scale
      type: kp_random_scale
      kwargs:
        scale_center: 1
        scale_factor: 0.3

random_rotate:

  .. code-block:: yaml

    random_rotate: &random_rotate
      type: kp_random_rotate
      kwargs:
        rotate_angle: 40

crop_square:

  .. code-block:: yaml

    crop_square: &crop_square
      type: kp_crop_square

flip:

  .. code-block:: yaml

    flip: &flip
      type: kp_flip
      kwargs:
        match_parts: [[1,2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

crop:

  .. code-block:: yaml

    crop_train: &crop
      type: kp_crop
      kwargs:
        means: [ 0.485,0.456,0.406 ]
        stds: [ 0.229,0.224,0.225 ]
        img_norm_factor: 255.0
        in_h: 256
        in_w: 192

label_trans:

  .. code-block:: yaml

    label_trans: &label_trans
      type: kp_label_trans
      kwargs:
        label_type: 'SCE'
        bg: True
        radius: [0,0,1,2]
        strides: [32,16,8,4]
        in_h: 256
        in_w: 192
        num_kpts: 17
