Release History
===============

v0.2.0
------

External dependency version
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Spring2 0.6.0 
* Cluster environment s0.3.3 / s0.3.4
* spring_aux-0.6.7.develop.2022_05_07t08_45.333adcd0-py3-none-any.whl

Breaking Changes
^^^^^^^^^^^^^^^^

* Structural composition of the second stage of detection is reconstructed, in order to carry out quantization and sparse training more conveniently.
* Modified model deployment config. Cfg can be inquired from here

  .. code-block:: bash
         
    # Deleted detector
    # Default config (for example, det)：
    to_kestrel:
        toks_type: det  # task type
        save_to: KESTREL  # model save path
        plugin: essos  # kestrel plugin
        ...

Highlights
^^^^^^^^^^

* Support 3D Point-Pillar series such as Pointpillar,Second, CenterPoint and so on
* Support Vision Transformer series such as Swin-Transformer, VIT，CSWin Transformer
* Support segmentation sota: Segformer，HrNet and high performance baseline
* Support newest detetion distillation and improve performance largely
* Support Sparse training
* Support quantify pipeline QAT and PTQ
* Support self-supervise pipeline such as MOCO, SimClr, simsiam, MAE
* Support Multi-task training
* Support detection, classification, segmentation, keypoint model deployment, and evaluting and publishing on Adela
* Support large dataset training and testing on multi-task (Rank dataset) and memory-friendly interface
* Support Chinese and English docs

New Features
^^^^^^^^^^^^

* Add Condinst FCOS
* Add ceph example `#16 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/16>`_
* Support task isolation by setting environment variable
* Add voxel generator to replace spconv interface
* Support ceph reading for kitti dataset
* Support multi-label and multi-classifier for class task
* Support evaluating multiple test datasets respectively `#8 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/8>`_
* Refactor Rank dataset to support training and inferencing in classification and detection
* Large dataset memory optimization
* Support time logger for every step in training
* Support Softer NMS
* Support torch Toonnx
* Add CI testing and deploying examples for all tasks `#29 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/29>`_
* Support mixup and cutmix augmentations for multicls task.
* Support bad case analyse for detection and classification `#36 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/36>`_

Bug Fixes
^^^^^^^^^

* Fix no registering stitch_expand
* Fix some typos bugs
* Fix memory bugs in spconv and numba
* Fix many packages debug logger bug
* Fix can't import InterpolationMode bug `#23 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/23>`_
* Fix swin and cswin shape bug for detection
* Fix base_multicls and roi_predictor bugs in condinst while return_pos_inds = True
* Fix ema model importing in inferencing
* Fix out planes bugs in swin
* Fix meta_file bug in cls_dataset 
* Fix fp16 grad clipping bug
* Fix syncbn bug in inferencing `#33 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/33>`_
* Fix finalize bug in single gpu testing
* Fix dist backend bug
* Avoid linklink initializing and dataset building in to_kestrel `#22 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/22>`_
* Adapt model deploying for model without postprocess
* Fix ema model loading for deployment
* Fix setting different class alpha for torch_sigmoid_focal_loss
* Support auto-saving best performance model foe kitti evaluator
* Add module prefix for loss `#19 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/19>`_
* Refacting adela interface without release.json
* Fix gdbp not supporting multi-bs input
* Support setting nart config for adela deployment `#44 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/44>`_
* Fix deploying bug in RetinaHead with IoU
* Fix loading environment variable bug in time logger `#57 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/57>`_

v0.1.0
-------

Hightlights
^^^^^^^^^^^^^^^^^^^^^

* Deployable high accuracy baselines, a complete model production process, and directly deploying and evaluting with Adela.
* Unified training task interface, which supports individual and joint training of detection, classification, key-point detection, and semantic segmentation tasks.
* Compatibility with the checkpoints of POD, Prototype, and other frameworks, making the transportation easy.
* Developing with Plugin mode, supporting custom modules.
* Simple model distillation methods.
* An unified training environment with simple training interfaces, allowing users to finish train by registering small number of modules.
* Unified file reading interfaces that support ceph, lustre, and other reading backends.
