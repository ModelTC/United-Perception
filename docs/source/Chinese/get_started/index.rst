快速上手/Get Started
====================

* :ref:`GSTrainAnchor`
* :ref:`EvalAnchor`
* :ref:`DemoAnchor`


使得用户能够在很短的时间内快速产出模型，掌握 UP 的使用方式。

.. _GSTrainAnchor:

Train
-----

Step1: 修改dataset 路径

  .. code-block:: bash

    dataset:
      type: coco # dataset type
        kwargs:
          source: train
          meta_file: coco/annotations/instances_train2017.json
          image_reader:
            type: fs_opencv
            kwargs:
              image_dir: coco/train2017
              color_mode: BGR
          transformer: [*flip, *train_resize, *to_tensor, *normalize]


Step2: 训练

    .. code-block:: bash

      ROOT=../
      T=`date +%m%d%H%M`
      export ROOT=$ROOT
      cfg=$2
      export PYTHONPATH=$ROOT:$PYTHONPATH
      python -m up train \
        --ng=$1 \
        --launch=pytorch \
        --config=$cfg \
        --display=10 \
        2>&1 | tee log.train.$T.$(basename $cfg)

      # ./dist_train.sh <num_gpu> <config>
      ./dist_train.sh 2 configs/det/yolox/yolox_tiny.yaml


Step3: FP16 设置以及其他一些额外的设置

  .. code-block:: yaml

    runtime:
      fp16: True # ddp 后端
      runner:
        type: base # 默认是base，也可以根据需求注册所需的runner，比如量化quant


.. _EvalAnchor:

Evaluate
--------

评测脚本, 现在将train test 合成了一个指定，在命令行指定 -e 即可启动测试

  .. code-block:: bash

    ROOT=../
    T=`date +%m%d%H%M`
    export ROOT=$ROOT
    cfg=$2
    export PYTHONPATH=$ROOT:$PYTHONPATH

    python -m up train \
      -e \
      --ng=$1  
      --launch=pytorch \
      --config=$cfg \
      --display=10 \
      2>&1 | tee log.test.$T.$(basename $cfg)

    # ./dist_test.sh <num_gpu> <config>
    ./dist_test.sh 2 configs/det/yolox/yolox_tiny.yaml

.. _DemoAnchor:

Demo
----

Step1: 修改cfg

  .. code-block:: bash

    runtime:
      inferencer:
        type: base
        kwargs:
          visualizer:
            type: plt
            kwargs:
              class_names: ['__background__', 'person'] # class names
              thresh: 0.5

Step2: inference

  .. code-block:: bash

    ROOT=../
    T=`date +%m%d%H%M`
    export ROOT=$ROOT
    cfg=$2

    python -m up inference \
      --ng=$1
      --launch=pytorch \
      --config=$cfg \
      2>&1 | tee log.inference.$T.$(basename $cfg)

    # ./dist_inference.sh  <num_gpu> <config>
    ./dist_inference.sh 1 configs/det/yolox/yolox_tiny.yaml

