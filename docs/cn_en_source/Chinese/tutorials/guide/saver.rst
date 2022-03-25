存储模块配置
===========

模型，日志等结果保存设置

.. warning::

   * auto_resume: 该字段为 True 时，自动从 save_dir 下 load 最新 checkpoint。
     优先级 auto_resume > opts > resume_model > pretrain_model

base
----

模型自动保存在save_dir目录下

.. code-block:: yaml

    saver: # Required.
        save_dir: checkpoints # dir to save checkpoints
        pretrain_model: /mnt/lustre/share/DSK/model_zoo/pytorch/imagenet/resnet50-19c8e357.pth
        results_dir: results_dir  # dir to save detection results. i.e., bboxes, masks, keypoints
        auto_resume: True

ceph
----

模型直接保存在ceph上

.. code-block:: yaml

    saver:
      type: ceph
      kwargs:
        save_cfg:
          save_dir: checkpoints
          ceph_dir: ceph-sh1424-det:s3://test/sh1424  # 前缀
          results_dir: results_dir
          pretrain_model: xxxx
          auto_resume: True

.. warning::

    * 在save_dir下会存储一个ckpt_ex.pth.ini，里面会存储一些link信息，包含ceph的存储的完整路径
    * ceph上的存储路径，$ceph_dir/$pwd/$save_dir/ckpt_xx.pth，会自动拼接当前的绝对路径
    * 当ceph出现bug（超时/配置错误等），会默认改为lustre存储，不会导致训练崩掉
