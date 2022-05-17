Get Started
===========

* :ref:`GSTrainAnchorEn`
* :ref:`EvalAnchorEn`
* :ref:`DemoAnchorEn`


Guiding users to produce models in a short time and master the using of UP.

.. _GSTrainAnchorEn:

Train
-----

Step1: change the path of dataset.

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


Step2: training

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


      #./dist_train.sh <num_gpu> <config>
      ./dist_train.sh 2 configs/det/yolox/yolox_tiny.yaml

Step3: the setting of FP16 and others.

  .. code-block:: yaml

    runtime:
      fp16: True # ddp backend
      runner:
        type: base # Default is base, or register the runner according to the requirement such as quant.


.. _EvalAnchorEn: 

Evaluate
--------

The evaluation script merges tesing into training where tesing can be started by assigned -e in the training order.

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
    ./dist_test.sh 1 configs/det/yolox/yolox_tiny.yaml

.. _DemoAnchorEn:

Demo
----

Step1: revise the config.

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

Step2: inference.

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

    # ./dist_inference.sh <num_gpu> <config>
    ./dist_inference.sh 1 configs/det/yolox/yolox_tiny.yaml
