3D detection
============

UP supports the whole pipline of training and interfering;

`Codes <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/up/tasks/det_3d>`_

Configs
-------

It contains the illustration of common configs.

`Repos <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/det_3d>`_

Dataset related modules
-----------------------

1. Dataset types:

  * kitti

2. The type of datasets can be chosen by setting 'type' in Dataset (default is kitti). The config is as followed.

  .. code-block:: yaml
    
    dataset:
      type: kitti
      kwargs:
        meta_file: /mnt/lustre/share/wangxiaodan/data/kitti/kitti_infos/kitti_infos_train.pkl
        class_names: *class_names
        get_item_list: &get_item_list ['points']
        training: True
        transformer: [*point_sampling, *point_flip,*point_rotation,*point_scaling, *to_voxel_train]
        image_reader:
          type: kitti
          kwargs:
            image_dir: /mnt/lustre/share/wangxiaodan/data/kitti/training/
            color_mode: None
