Saving config
=============

The config of saving models and logs.

.. warning::

   * auto_resume: automatically download the latest checkpoint from save_dir when auto_resume is True.
   * Priority: 'auto_resume' > 'opts' > 'resume_model' > 'pretrain_model'.

base
----

Models are automatically saved in the 'save_dir'.

.. code-block:: yaml

    saver: # Required.
        save_dir: checkpoints # dir to save checkpoints
        pretrain_model: /mnt/lustre/share/DSK/model_zoo/pytorch/imagenet/resnet50-19c8e357.pth
        results_dir: results_dir  # dir to save detection results. i.e., bboxes, masks, keypoints
        auto_resume: True

ceph
----

Models are automatically saved in the 'ceph'.

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

    * 'ckpt_ex.pth.ini' will be saved in 'save_dir' which saves some link information such as the absolute saving path of 'ceph'.
    * the saving path of 'ceph', $ceph_dir/$pwd/$save_dir/ckpt_xx.pth, will automatically join the current absolute path.
    * saving will be executed in lustre when 'ceph' has any bug, keeping the running of training.
