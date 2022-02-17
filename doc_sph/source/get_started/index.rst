快速上手/Get Started
====================

* :ref:`GSTrainAnchor`
* :ref:`EvalAnchor`
* :ref:`DemoAnchor`
* :ref:`DeployAnchor`


使得用户能够在很短的时间内快速产出模型，掌握 UP 的使用方式。

.. _GSTrainAnchor:

Train
-----

Step1: 修改dataset 路径，基本格式沿袭POD的格式，可以参考POD的文档 

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


Step2: 训练
  1. Srun 直接启动脚本

    .. code-block:: bash

      g=$(($2<8?$2:8))
      srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
          --job-name=$cfg \
      python -m up train \
        --config=$cfg \
        --display=1 \
        --backend=linklink \ # 默认后端，支持dist (ddp) 和 linklink 2种
        2>&1 | tee log.train

      # ./train.sh <PARTITION> <num_gpu> <config>
      ./train.sh ToolChain 8 configs/det/yolox/yolox_tiny.yaml

  2. Spring.submit 启动脚本

    .. code-block:: bash

      export ROOT=$ROOT
      cfg=$2
      export PYTHONPATH=$ROOT:$PYTHONPATH
      CPUS_PER_TASK=${CPUS_PER_TASK:-4}

      spring.submit run -n$1 -p spring_scheduler --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
      "python -m up train \
        --config=$cfg \
        --display=10 \
        --backend=linklink \
        2>&1 | tee log.train "

      #./train.sh <num_gpu> <PARTITION> <config> <job-name>
      ./train.sh 8 ToolChain configs/det/yolox/yolox_tiny.yaml yolox_tiny

    
Step3: FP16 设置以及其他一些额外的设置

  .. code-block:: bash

    runtime:
      fp16: # linklink 后端
          keep_batchnorm_fp32: True
          scale_factor: dynamic
      # fp16: True # ddp 后端
      runner:
        type: base # 默认是base，也可以根据需求注册所需的runner，比如量化quant


.. _EvalAnchor: 

Evaluate
--------

评测脚本, 沿袭POD的模式，现在将train test 合成了一个指定，在命令行指定 -e 即可启动测试

  .. code-block:: bash

    g=$(($2<8?$2:8))
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
        --job-name=$cfg \
    python -m up train \
      -e \
      --config=$3 \
      --display=1 \
      2>&1 | tee log.eval

    # ./eval.sh <PARTITION> <num_gpu> <config>
    ./eval.sh ToolChain 1 configs/det/yolox/yolox_tiny.yaml

.. _DemoAnchor:

Demo
----

Step1: 修改cfg，沿袭POD的格式 

  .. code-block:: bash

    inference:
      visualizer:
        type: plt
        kwargs:
          class_names: ['__background__', 'person'] # class names
          thresh: 0.5

Step2: inference

  .. code-block:: bash

    g=$(($2<8?$2:8))
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
        --job-name=$3 \
    python -m up inference \
      --config=$3 \\
      -i imgs \
      -v vis_dir \
      2>&1 | tee log.inference

    # ./inference.sh <PARTITION> <num_gpu> <config>
    ./inference.sh ToolChain 1 configs/det/yolox/yolox_tiny.yaml


.. _DeployAnchor:

Deploy
-------

UP 支持将检测模型转化为caffemodel格式

  .. code-block:: bash

    #!/bin/bash

    ROOT=../
    T=`date +%m%d%H%M`
    export ROOT=$ROOT
    cfg=$2
    export PYTHONPATH=$ROOT:$PYTHONPATH
    CPUS_PER_TASK=${CPUS_PER_TASK:-4}

    spring.submit run -n$1 -p spring_scheduler --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
    "python -m up to_caffe \
      --config=$cfg \
      --save_prefix=tocaffe \
      --input_size=3x512x512 \
      --backend=linklink \
      2>&1 | tee log.tocaffe.$T.$(basename $cfg) "

UP 支持将检测模型转化为kestrel格式

  .. code-block:: bash

    ROOT=../
    T=`date +%m%d%H%M`
    export ROOT=$ROOT
    cfg=$2
    export PYTHONPATH=$ROOT:$PYTHONPATH
    CPUS_PER_TASK=${CPUS_PER_TASK:-4}

    spring.submit run -n$1 -p spring_scheduler --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
    "python -m up to_kestrel \
      --config=$cfg \
      --save_to=kestrel_model \
      2>&1 | tee log.tokestrel.$T.$(basename $cfg) "

部署例子:

检测配置：
`Detection deploy configs <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/tree/master/configs/det/deploy>`_

分类配置：
`Classification deploy configs <https://gitlab.bj.sensetime.com/spring2/universal-perception/-/tree/master/configs/cls/deploy>`_
