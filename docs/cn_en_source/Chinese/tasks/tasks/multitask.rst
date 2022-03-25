多任务联合训练
==============

UP支持多任务的联合训练，推理; 训练流程为每个task分支网络处理对应的训练数据，计算loss，之后将各任务loss计算总和并反向传播更新梯度
`具体代码 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/up/tasks/multitask>`_

配置文件
--------

`配置文件 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/multitask>`_
其中包括示例配置文件

配置文件示例
------------

具体配置细节可以参考上述配置文件仓库，以下为必要设置示例:

  .. code-block:: yaml

    # runner: 修改runner
    runtime:
      runner:
        type: multitask
      task_names: &task_names [det, cls]
      ...

    # dataset部分需要写出所需task的train/test，必须包含train字段（作为主task，epoch等按照主task来计算）
    dataset: 
      train: ...
      test: ...
      train_cls: ...
      test_cls: ...

    # multitask cfg
    multitask_cfg:
      notions:
        det: &det 0  # 后续用引用代替数字，更稳妥
        cls: &cls 1
      task_names: *task_names  # task名称
      datasets: # 指定每个task的训练/测试数据集
        train: [train, train_cls]  
        test: [test, test_cls]
      debug: &multitask_debug false

    # net： 需要对每个head做一些特殊的操作，方便控制流程
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
                idxs: [ *det ]  # 主要通过这个idxs控制，此处表明这个head只会在det数据时forward
                debug: *multitask_debug
