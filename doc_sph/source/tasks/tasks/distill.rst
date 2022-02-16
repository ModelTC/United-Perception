知识蒸馏
========

UP支持模型蒸馏训练; 具体流程为通过蒸馏teacher模型与student模型的若干个特征图，提升student模型性能
`具体代码 <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/tree/dev/up/tasks/distill>`_


子任务配置文件示例
------------------

* `classification <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/resnet/res18_kd.yaml>`_
* `detection <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x_mimic.yaml>`_

配置文件说明
------------

以下为配置文件示例:

  .. code-block:: yaml

    # runner: 修改runner
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
