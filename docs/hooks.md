# Hooks
Hooks are used to monitor the training process, including timing, log, visualization and so on.
The details can be found in [hooks.py](eod/utils/general/hook_helper.py).

## Typical hooks
All types of hooks inherit from **Hook** classes. EOD supports:
* train_val_logger
* auto_checkpoint
* grad_clipper
* auto_save_best
* reload

### TrainValLogger
It is used to output training logs, including printing loss, time consumption, remaining time, etc. 
During the training, EOD will maintain the training log with tensorboard, including loss, accuracy, etc.

```yaml
hooks:
  - type: train_val_logger
    kwargs:
      freq: 2           # print log frequency
      skip_first_k: 5   # ignore 5 epochs time consume
      logdir: log       # tennsorboard log path
      summary_writer: tensorboard # choices = [tensorboard, pavi] # when use pavi, can not check log with tensorboard
```

### AutoSaveBest
Save the checkpoint with the best evaluating result.
```yaml
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
```

### AutoCheckpoint
The checkpoint is automatically saved when the experiment is killed.
```yaml
hooks:
  - type: auto_checkpoint
```

### Gradient Clip
Gradient clipping supports three modes: 
* Predefined norm 
* Averaged norm
* Moving averaged norm

```yaml
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
```

