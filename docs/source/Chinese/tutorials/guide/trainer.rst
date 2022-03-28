训练器配置
==========

该部分用于控制训练过程，包括warmup策略，优化算法，学习率调整等

.. note::

  * warmup 为逐步放大学习率以平稳训练过程的算法, UP提供了多种warmup方式："exp", "linear", "no_scale_lr"
  * 如果使用 warmup, warmup 初始学习率等于 base_lr * warmup_ratio, warmup 过程结束后的学习率为 base_lr * total_batch_size, total_batch_size 等于 batch size * gpu 的数量 
  * 配置文件 config 中的 lr 为 batch 中单张图像的学习率，如lr=0.00125，使用 8 张卡(每张 batch size 为 2)的情况下，最终学习率等于0.00125*16=0.02
  * 在配置文件 config 中设置 warmup_epochs 或者 warmup_iter 以开启 warmup 过程，warmup_epochs 等同于 warmup_iter，warmup_epochs 会自动转换为 warmup_iter
  * only_save_latest参数支持仅保存上一阶段的模型且使用时会使save_freq失效

.. code-block:: yaml

    trainer: # Required.
        max_epoch: 14 # total epochs for the training
        test_freq: 14 # test every 14 epochs (当大于max_epoch，则只在训练结束时进行测试）
        save_freq: 1 # 模型保存的epoch间隔
        # only_save_latest: False # 如果是True，仅保存上一阶段的模型且save_freq失效
        optimizer:
            type: SGD
            kwargs:
                lr: 0.00125
                momentum: 0.9
                weight_decay: 0.0001
        lr_scheduler: # lr_scheduler = MultStepLR(optimizer, milestones=[9,14],gamma=0.1)
            warmup_epochs: 1 # set to be 0 to disable warmup.
            # warmup_type: exp
            type: MultiStepLR
            kwargs:
                milestones: [9,12] # epochs to decay lr
                gamma: 0.1 # decay rate
