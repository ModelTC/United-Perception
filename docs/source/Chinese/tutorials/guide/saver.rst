存储模块配置
============

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
        pretrain_model: # pretrain_path 
        results_dir: results_dir  # dir to save detection results. i.e., bboxes, masks, keypoints
        auto_resume: True

