Hooks
=====

Hooks are used to monitor the training process including time, log, visualization, and so on. Details can be found in 'hooks.py'.

Common hooks
------------

All classes of hook inherit from the 'Hook' class. UP supports the following classes.

* train_val_logger
* auto_checkpoint
* grad_clipper
* auto_save_best
* reload
* memory_checkpoint

TrainValLogger
--------------

'TrainValLogger' is used to output the training log which prints losses, time consuming, time remaining, and so on.
UP keeps the traing log which contains tensorboard in training. The log contains losses, accuracy, and so on.

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

Keeping the checkpoint with highest accuracy.

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

The checkpoint will be automatically saved when training is stopped.

  .. code-block:: yaml
    
    hooks:
      - type: auto_checkpoint

Gradient Clip
-------------

Gradient clipping supports the following three modes.

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

Dynamic checkpoint is a video memory optimization trick, which can be applied to the following situations

    * The input size of model training can be quantified
    * The input size of the model is the same, and the video memory occupation of the model is the same (or similar)
    * The input size of the model changes

1. Modify the hook field in the configuration file

  .. code-block:: yaml

    hooks:
      - type: memory_checkpoint
        kwargs:
            enable: True
            checkpoint_patterns:
              backbone:
                  patterns_mode: level
                  level:
                    num: 4 #Set to 2 when ResNet
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
              warmup_iters: 30 # The iterations that control the video memory occupation of the profiling model, the more it set, the more information about the video memory occupation is collected, and the more accurate the prediction model is
              max_memory: 8 # Control the upper limit of video memory usage (torch. CUDA. Memory_allocated) (GB) of DC
              debug_freq: 10 # Frequency of printing information
              strategy: greedy # memory_time or greedy

  * warmup_iters

    * The unit is iteration, which controls the iterations of the video memory occupation of the profiling model. The more settings are set, the more information about the video memory occupation is collected, and the more accurate the prediction model is. Profiling also consumes a lot of time, but the overhead is less than warmup_iters * iter_time
    * If it is a classification task, the video memory occupation has not changed, you can lower warmup_iters, such as 10.
    * It can be set to about 30 if it is a task with obvious relationship between input_size and video memory occupation.
    * If there are large changes in video memory under the same input, this task is not applicable in theory. If you use dynamic checkpoint, you need to set it to more values.

  * max_memory

    * The unit is GB, which is the upper limit of the video memory usage (torch. CUDA. Memory_allocated) (GB) of the control DC
    * When the task is a fixed input classification task, memory_threshold can be set higher.
    * If the task is a detection task of 2 stages, you need to lower the memory_threshold。 Because this type of task changes greatly when the same input exists, it needs to use more conservative settings.
    * If oom occurs during task execution, if it is affected by video memory fragments, you need to lower the memory_threshold, such as 1 GB or 0.5 GB; If the video memory optimization space of the backbone is insufficient, it may be necessary to replace other methods for optimization.

  * debug_freq

    * After warmup_iters, the frequency of output optimization schedule

  * strategy

    * You can choose memory_time or greedy

2. Main Process

  In UP, it is managed by the DynamicCheckpointManager class, and the execution state of the current model is judged by the function interfaces such as before_forward, and relevant information is collected. The specific process is as follows:

  * step1    Collect video memory data, about 10 ~ 30 iters

    * before_forward: Record the current input size and video memory occupation, and reset the pytorch video memory statistics; Get the module set that should be checked at present (default is all Bottleeck, SwinTransformerBlock, Encoder, etc.)
    * after_update: Record the maximum video memory occupation, and calculate the video memory size required by the model from the beginning of forward to the end of update
  
  * step2    Optimize video memory occupation

    * before_forward: Record the current input size. If there is an optimization plan for this input in the cache (the input size will be round merged), the plan will be directly applied; On the contrary, the required checkpoint module set is generated by greedy or other algorithms, used as an optimization plan for application, and saved in the cache
    * dc_cast_forward(Not in Manager): Check whether the current module is in the checkpoint module set. If it is, execute checkpoint forward; Otherwise, forward is executed normally.


