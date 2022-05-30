3D检测
======

UP支持3D检测任务训练、推理的全部流程;
`具体代码 <https://github.com/ModelTC/EOD/tree/main/up/tasks/det_3d>`_

配置文件
--------

`代码仓库 <https://github.com/ModelTC/EOD/tree/main/configs/det_3d>`_
其中包括常用算法配置文件

数据集相关模块
--------------

1. 数据集类型包括:

  * kitti

2. 数据集类型通过设置Dataset的type来选择，默认为kitti，配置文件示例如下:

  .. code-block:: yaml
    
    dataset:
      type: kitti
      kwargs:
        meta_file: kitti/kitti_infos/kitti_infos_train.pkl
        class_names: *class_names
        get_item_list: &get_item_list ['points']
        training: True
        transformer: [*point_sampling, *point_flip,*point_rotation,*point_scaling, *to_voxel_train]
        image_reader:
          type: kitti
          kwargs:
            image_dir: kitti/training/
            color_mode: None
