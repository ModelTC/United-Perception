Trainer
=======

This part introduces how to control the training including warmup strategies, optimization algorithms, adjusting learning rate, and so on. 

.. note::

  * warmup is used to gradually raise the learning rate for smoothing the training, which contains 'exp', 'linear', and 'no_scale_lr'.
  * The initial warmup learning rate equals 'base_lr' * 'warmup_ratio'. The learning rate equals 'base_lr' * 'total_batch_size' after warmup, and the 'total_batch_size' equals 'batch size' * 'gpu'.  
  * 'lr' in config is the learning rate of a single image. For example, using 8 GPUs with 2 batches and lr=0.00125 makes the final learning rate equal 0.00125 * 16 = 0.02.
  * Starting warmup by setting 'warmup_epochs' or 'warmup_iter' in the config. 'warmup_epochs' will be transformed into 'warmup_iter'.
  * 'only_save_latest' supports only keeping the latest model and will invalidate 'save_freq'. 

.. code-block:: yaml

    trainer: # Required.
        max_epoch: 14 # total epochs for the training
        test_freq: 14 # test every 14 epochs (Only tesing after all training when it is larger than max_epoch）
        save_freq: 1 # save model every save_freq epoches.
        # only_save_latest: False # if True, only keep the latest model and invalidate save_freq.
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
