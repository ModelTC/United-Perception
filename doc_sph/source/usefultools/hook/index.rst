运行时钩子
==========

运行时钩子被用来监测训练过程，包括时间（timing），日志（log），可视化（visualization）等。
细节可以在 hooks.py 中找到。

常用类别
--------

所有钩子类都是从 Hook 类中被继承。UP 支持以下类：

* train_val_logger
* auto_checkpoint
* grad_clipper
* auto_save_best
* reload

TrainValLogger
--------------

TrainValLogger 被用来输出训练日志，包括打印损失值，时间消耗，剩余时间等。
在训练中，UP 会保留含有 tensorboard 的训练日志，包括损失值和准确率。

  .. code-block:: yaml
    
    hooks:
      - type: train_val_logger
        kwargs:
          freq: 2           # print log frequency
          skip_first_k: 5   # ignore 5 epochs time consume
          logdir: log       # tennsorboard log path
          summary_writer: tensorboard # choices = [tensorboard, pavi] # when use pavi, can not check log with tensorboard

AutoSaveBest
------------

保留具有最高精度的 checkpoint.

  .. code-block:: yaml
    
    trainer:
      max_epoch: 19              # total epoch nums
      test_freq: 19              # (best ckpt) validate interval
      test_start: 1              # (best ckpt) validate start epoch
      save_freq: 1               # ckpt save interval
      save_start: 1              # ckpt save start epoch
      save_cur_start: -1         # default -1，if the num> 0 means ckpt the epoch start to real-time save 
      save_cur_freq: 2           # ckpt real-time save interval(epochs)

    hooks:
      - type: auto_save_best

AutoCheckpoint
--------------

当训练被中止时，checkpoint 是被自动保存的。

  .. code-block:: yaml
    
    hooks:
      - type: auto_checkpoint

Gradient Clip
-------------

梯度剪切支持以下三种模式：

    * Predefined norm
    * Averaged norm
    * Moving averaged norm

  .. code-block:: yaml
    
    hooks:
      - type: grad_clipper
        kwargs:
          mode: pre_defined  # specific norm
          norm_type: 2
          max_norm: 10

    or

    hooks:
      - type: grad_clipper
        kwargs:
          mode: average    # average
          norm_type: 2
          tolerance: 2.0   # if over average, 2 times clip

    or

    hooks:
      - type: grad_clipper
        kwargs:
          mode: moving_average  # sliding average
          momentum: 0.9
          norm_type: 2
          tolerance: 5.0        # if over average, 2 times clip




    
