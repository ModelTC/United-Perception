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
* memory_checkpoint

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

MemoryCheckpoint
-----------------

dynamic checkpoint 是显存优化trick，其可应用于以下情况

    * 模型训练的输入大小可量化
    * 模型输入大小相同，模型的显存占用也相同（或相近）
    * 模型输入大小存在变化

1. 修改配置文件中hook字段

  .. code-block:: yaml

    hooks:
      - type: memory_checkpoint
        kwargs:
            enable: True
            checkpoint_patterns:
              backbone:
                  patterns_mode: level
                  level:
                    num: 4 #当resnet时设置为2
              neck:
                  patterns_mode: level
                  level:
                    num: 1
              roi_head:
                  patterns_mode: level
                  level:
                    num: 1
                  share_weight_num: 5
            dc_cfg:
              warmup_iters: 30 #控制profiling 模型显存占用的 iterations，设置的越多，收集到的显存占用信息也越多，预测模型也越准确
              max_memory: 8 #控制DC的显存用量(torch.cuda.memory_allocated())(GB)上限
              debug_freq: 10 #打印信息的频率
              strategy: greedy # memory_time or greedy

  * warmup_iters

    * 单位 iteration, 控制profiling 模型显存占用的 iterations，设置的越多，收集到的显存占用信息也越多，预测模型也越准确。profiling也会消耗大量的时间，但是overhead 小于 warmup_iters * iter_time
    * 如果是分类任务，显存占用没有变化，可以将warmup_iters调低，比如10。
    * 如果是 input size - 显存占用关系比较明显的任务，则可以设为30左右。
    * 如果是相同 input size 下也有较大的显存变化，那么理论上不适用该任务。如果使用 dynamic checkpoint，则需要将其设置更多的值。

  * max_memory

    * 单位GB, 这部分为控制DC的显存用量(torch.cuda.memory_allocated)(GB)上限
    * 当任务为输入固定的分类任务时，memory_threshold 可以调的更高。
    * 如果任务为2 stage的目标检测任务，则需要调低 memory_threshold。因为该类任务显存在相同输入的情况下， 变化较大， 所以需要使用更加保守的设置。
    * 如果任务执行过程中发生了 OOM，如果是显存碎片的影响，则需要调低 memory_threshold，如 1 GB 或 0.5 GB为单位；如果是backbone本身显存优化空间不足，那么可能需要替换其它方法进行优化。

  * debug_freq

    * 在warmup_iters 后，输出优化schedule的iter频率

  * strategy

    * 可选memory_time或者greedy

2. 主要流程

  在up中通过DynamicCheckpointManager类进行管理，通过before_forward等函数接口来判断当前模型执行状态，并收集相关信息。其具体流程如下：

  * step1 收集显存数据，约10~30 iters

    * before_forward: 记录当前input输入大小、显存占用，并重置pytorch显存统计数据；获得当前应该checkpoint的Module集合（默认是全部的Bottleneck、SwinTransformerBlock、Encoder等）
    * after_update：记录最大的显存占用，并计算出 model 从 forward 开始到 update 结束所需要的显存大小
  
  * step2 优化显存占用

    * before_forward：记录当前 input 输入大小，若 cache 中已有该 input（input 大小会经过进行 round 合并 ） 的优化plan，则直接应用该 plan；反之，则通过 Greedy 或者其它算法生成所需要 checkpoint module set，以此作为优化 plan 进行应用，并保存到 cache 中
    * dc_cast_forward（不在 manager 中）：查找当前 Module 是否在 checkpoint module set 中，若在，则执行 checkpoint forward；反之，则正常执行 forward。

