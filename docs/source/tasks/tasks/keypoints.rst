关键点检测
==========

UP支持关键点检测任务训练、推理的全部流程;
`具体代码 <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/tree/master/up/tasks/kp>`_

配置文件
--------

`代码仓库 <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/tree/master/configs/kp>`_
其中包括常用算法配置文件

数据集相关模块
--------------

1. 数据集类型包括:

  * keypoint

数据预处理
----------

UP 的关键点检测模块提供了额外的数据增广函数，包括: keep_aspect_ratio, random_scale等; 配置细节如下所示:

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
