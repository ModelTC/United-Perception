Knowledge distillation
======================

UP supports knowledge distillation. The specific pipline is improving the student model by distilling multiple feature maps between teacher models and student models.

`Codes <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/up/tasks/distill>`_


Config of sub-task
------------------

* `classification <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/master/configs/cls/resnet/res18_kd.yaml>`_
* `detection <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/master/configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x_mimic.yaml>`_

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
      mimic_name: job1
      mimic_type: kl   # mimic loss
      loss_weight: 1.0
      teacher:
        teacher_weight: xxx   # path of teacher model
        mimic_name: [xxx]  # teacher mimic feature name
      student:
        mimic_name: [xxx]  # student mimic feature name

    ...

    # teacher model cfg
    teacher:
      - name: backbone
        type: resnet50
        kwargs:
      ...
      
    # student model cfg
    net:
      ...
