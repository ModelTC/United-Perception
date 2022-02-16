分类
====

UP支持分类任务训练、部署、推理的全部流程;
`具体代码 <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/tree/dev/up/tasks/cls>`_

配置文件
--------

`代码仓库 <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/tree/master/configs/cls>`_
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
        meta_file: /mnt/lustre/share/images/meta/train.txt
        image_reader:
           type: fs_pillow
           kwargs:
             image_dir: /mnt/lustre/share/images/train
             color_mode: RGB
        transformer: [*random_resized_crop, *random_horizontal_flip, *pil_color_jitter, *to_tensor, *normalize]

部署模块
--------

转换kestrel模型时，需要使用CLSToKestrel，具体配置如下:

  .. code-block:: yaml

    to_kestrel:
      toks_type: cls   # 通过设置toks_type
      model_name: Res50
      add_softmax: False
      pixel_means: [123.675, 116.28, 103.53]
      pixel_stds: [58.395, 57.12, 57.375]
      is_rgb: True
      save_all_label: True
      type: 'UNKNOWN'
