Multi-task joint training
=========================

UP supports multi-task joint training and inference. The training pipeline firstly gives each task the corresponding training data, secondly computes loss, thirdly sums up the losses, and finally backwards for updating gradients.

`Codes <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/up/tasks/multitask>`_

Configs
-------

It contains the illustrtion of common configs.

`Configs <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/multitask>`_

Illustration of configs
-----------------------

The details can be refered to the above configs. The necessary setting is as followed.

  .. code-block:: yaml

    # runner: revising runner
    runtime:
      runner:
        type: multitask
      task_names: &task_names [det, cls]
      ...

    # Train/test of the task should be set in the datasetpart, which must contain 'train'.
    dataset: 
      train: ...
      test: ...
      train_cls: ...
      test_cls: ...

    # multitask cfg
    multitask_cfg:
      notions:
        det: &det 0  # will be further replaced by numbers. 
        cls: &cls 1
      task_names: *task_names  # task name.
      datasets: # training/testing dataset for every task.
        train: [train, train_cls]  
        test: [test, test_cls]
      debug: &multitask_debug false

    # net： every head needs special operations for easily controlling the pipline. 
      - name: roi_head
        prev: neck
        type: RetinaHeadWithBN
        kwargs:
          ...
        wrappers:
          - &det_wrapper
            type: multitask_wrapper
            kwargs:
              cfg:
                idxs: [ *det ]  # controlling the forwarding task
                debug: *multitask_debug
