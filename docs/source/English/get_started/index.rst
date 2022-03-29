Get Started
===========

* :ref:`GSTrainAnchor`
* :ref:`EvalAnchor`
* :ref:`DemoAnchor`
* :ref:`DeployAnchor`


Guiding users to produce models in a short time and master the using of UP.

.. _GSTrainAnchor:

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
  1. Directly runing the shell of Srun.

    .. code-block:: bash

      cfg=$3

      g=$(($2<8?$2:8))
      srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
          --job-name=$cfg \
      python -m up train \
        --config=$cfg \
        --display=1 \
        --backend=linklink \ # the default backend which supports dist (ddp) and linklink.
        2>&1 | tee log.train

      # ./train.sh <PARTITION> <num_gpu> <config>
      ./train.sh ToolChain 8 configs/det/yolox/yolox_tiny.yaml

  2. Directly runing the shell of Spring.submit.

    .. code-block:: bash

      export ROOT=$ROOT
      cfg=$3
      export PYTHONPATH=$ROOT:$PYTHONPATH
      CPUS_PER_TASK=${CPUS_PER_TASK:-4}

      spring.submit run -p $1 -n$2 --gpu --job-name=$cfg --cpus-per-task=${CPUS_PER_TASK} \
      "python -m up train \
        --config=$cfg \
        --display=10 \
        --backend=linklink \
        2>&1 | tee log.train "

      #./train.sh <PARTITION> <num_gpu> <config>
      ./train.sh ToolChain 8 configs/det/yolox/yolox_tiny.yaml

    
Step3: the setting of FP16 and others.

  .. code-block:: bash

    runtime:
      fp16: # linklink backend
          keep_batchnorm_fp32: True
          scale_factor: dynamic
      # fp16: True # ddp backend
      runner:
        type: base # Default is base, or register the runner according to the requirement such as quant.


.. _EvalAnchor: 

Evaluate
--------

The evaluation script merges tesing into training where tesing can be started by assigned -e in the training order.

  .. code-block:: bash

    cfg=$3

    g=$(($2<8?$2:8))
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
        --job-name=$cfg \
    python -m up train \
      -e \
      --config=$cfg \
      --display=1 \
      2>&1 | tee log.eval

    # ./eval.sh <PARTITION> <num_gpu> <config>
    ./eval.sh ToolChain 1 configs/det/yolox/yolox_tiny.yaml

.. _DemoAnchor:

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

    cfg=$3

    g=$(($2<8?$2:8))
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
        --job-name=$cfg \
    python -m up inference \
      --config=$cfg \
      -i=imgs \
      -v=vis_dir \
      -c=ckpt \
      2>&1 | tee log.inference

    # ./inference.sh <PARTITION> <num_gpu> <config>
    ./inference.sh ToolChain 1 configs/det/yolox/yolox_tiny.yaml


.. _DeployAnchor:

Deploy
-------

'to_caffe': UP supports tranforming the model to the caffemodel format.

  .. code-block:: bash

    #!/bin/bash

    ROOT=../
    T=`date +%m%d%H%M`
    export ROOT=$ROOT
    cfg=$3
    export PYTHONPATH=$ROOT:$PYTHONPATH
    CPUS_PER_TASK=${CPUS_PER_TASK:-4}

    spring.submit run -p $1 -n$2 --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
    "python -m up to_caffe \
      --config=$cfg \
      --save_prefix=tocaffe \
      --input_size=3x512x512 \
      --backend=linklink \
      2>&1 | tee log.tocaffe.$T.$(basename $cfg) "

'to_kestrel': UP supports tranforming the model to the kestrel format.

  .. code-block:: bash

    ROOT=../
    T=`date +%m%d%H%M`
    export ROOT=$ROOT
    cfg=$3
    export PYTHONPATH=$ROOT:$PYTHONPATH
    CPUS_PER_TASK=${CPUS_PER_TASK:-4}

    spring.submit run -p $1 -n$2 --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
    "python -m up to_kestrel \
      --config=$cfg \
      --save_to=kestrel_model \
      2>&1 | tee log.tokestrel.$T.$(basename $cfg) "

  .. note::

    * 'to_kestrel' needs adding the corresponding parameters in configs;
    * The parameters of specific sub-task which need to be added can refer to :ref:`tasks`. 

Deploy instance:

Detection：
`Detection deploy configs <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/det/deploy>`_

Classification：
`Classification deploy configs <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/cls/deploy>`_
