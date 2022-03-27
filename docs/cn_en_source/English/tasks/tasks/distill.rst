Knowledge distillation
======================

UP supports knowledge distillation. The specific pipline is improving the student model by distilling multiple feature maps between teacher models and student models.

`Codes <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/mimic/up/tasks/distill>`_


Config of sub-task
------------------

* `classification <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/mimic/configs/cls/resnet/res18_kd.yaml>`_
* `detection <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/mimic/configs/det/distill/faster_rcnn_r152_50_1x_feature_mimic.yaml>`_

Configs
-------

The config is as followed.

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

An advanced usage of the distillation is to mimic one student with multi teachers by multi methods. The config is as followed.

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
