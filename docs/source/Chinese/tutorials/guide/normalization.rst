数据归一化配置
==============

normalization支持五种模式

* freeze_bn： 固定mean和var

  .. code-block:: yaml

    normalize:
      type: freeze_bn

* solo_bn：单卡统计mean和var，不同步

  .. code-block:: yaml

    normalize:
      type: solo_bn

* pt_sync_bn：pytorch 多卡同步mean和var

  .. code-block:: yaml

    normalize:
      type: pt_sync_bn
      kwargs:
        group_size: 8

* gn: Group Normalization

  .. code-block:: yaml

    normalize:
      type: gn
      kwargs:
        num_groups: 32

* caffe_freeze_bn: 使用从caffe预加载的frozen bn

  .. code-block:: yaml

    normalize:
      type: caffe_freeze_bn
