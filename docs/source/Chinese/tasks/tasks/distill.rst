知识蒸馏
========

UP支持模型蒸馏训练; 具体流程为通过蒸馏teacher模型与student模型的若干个特征图，提升student模型性能
`具体代码 <https://github.com/ModelTC/EOD/-/tree/dev/up/tasks/distill>`_


子任务配置文件示例
------------------

* `classification <https://github.com/ModelTC/EOD/-/blob/dev/configs/cls/resnet/res18_kd.yaml>`_
* `detection <https://github.com/ModelTC/EOD/-/blob/dev/configs/det/distill/faster_rcnn_r152_50_1x_feature_mimic.yaml>`_

配置文件说明
------------

以下为配置文件示例:

  .. code-block:: yaml

    # runner: revise runner
    runtime:
      runner:
        type: kd
      ...

    # mimic config
    mimic:
      mimic_ins_type: base    # by default (optional)
      loss_weight: 1.0
      warm_up_iters: -1       # mimic loss warmup iters, better to align with the warmup_iter in lr_scheduler
      cfgs:                   # loss settings and some hyper params
        loss:                   # loss name
          type: kl_loss         # loss func
          kwargs:
            loss_weight: 1.0    # loss configs
      teacher0:                # teacher name, need to align with the teacher net config and start with the teacher prefix
        teacher_weight: xxx   # path of teacher model
        mimic_name: [xxx]  # teacher mimic feature name
      student:
        mimic_name: [xxx]  # student mimic feature name

    ...

    # teacher model cfg
    teacher0:
      - name: backbone
        type: resnet50
        kwargs:
      ...

    # student model cfg
    net:
      ...

UP蒸馏的进阶用法为:可以用多种mimic方法，从多个teacher中，对同一个studnet进行蒸馏。配置文件如下。

  .. code-block:: yaml

    # runner: revise runner
    runtime:
      runner:
        type: kd
      ...

    # mimic config
    mimic:
      mimic_as_jobs: True
      mimic_job0:           # the first mimic job, start with the mimic_job prefix
        mimic_ins_type: XXX
        loss_weight: 1.0
        warm_up_iters: -1       # mimic loss warmup iters, better to align with the warmup_iter in lr_scheduler
        cfgs:                 # loss settings and some hyper params
          ...
        teacher0:                # teacher name, need to align with the teacher net config and start with the teacher prefix
          teacher_weight: xxx   # path of teacher model
          mimic_name: [xxx]  # teacher mimic feature name
        student:
          mimic_name: [xxx]  # student mimic feature name
      mimic_job1:           # the second mimic job
        mimic_ins_type: XXX
        loss_weight: 1.0
        warm_up_iters: -1
        cfgs:
          ...
        teacher1:           # teacher names are unique
          teacher_weight: xxx
          mimic_name: [xxx]
        teacher2:           # one mimic job also supprot mimic from multi teachers
          teacher_weight: xxx
          mimic_name: [xxx]
        student:
          mimic_name: [xxx]


    ...

    # teacher model cfg
    teacher0:
      - name: backbone
        type: resnet50
        kwargs:
      ...

    teacher1:
      ...

    teacher2:
      ...
      
    # student model cfg
    net:
      ...
