# UP22222

![image](up-logo.png)

United Perception

**UP** (**U**nited **P**erception) is a general united-perception model production framework.
It aim on provide two key feature about Object Detection:

+ Efficient: we will focus on training **VERY HIGH ACCURARY** single-shot detection model, and model compression (quantization/sparsity) will be well addressed. 
+ Easy: easy to use, easy to add new features(backbone/head/neck), easy to deploy.
+ Large-Scale Dataset Training [Detail](https://github.com/ModelTC/rank_dataset)
+ Equalized Focal Loss for Dense Long-Tailed Object Detection [EFL](docs/equalized_focal_loss.md)
+ Improve-YOLOX [YOLOX-RET](docs/benchmark.md)
+ Quantization Aware Training(QAT) interface based on [MQBench](https://github.com/ModelTC/MQBench).


The master branch works with **PyTorch 1.8.1**.
Due to the pytorch version, it can not well support the 30 series graphics card hardware.

## Install
```shell
source s0.3.4
./easy_setup.sh <partition>
```

## Get Started
Some example scripts are supported in scripts/.

### Export Module
Export up into ROOT and PYTHONPATH

```shell
ROOT=../../
export ROOT=$ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
```

### Train
Step1: edit meta_file and image_dir of image_reader:

```yaml
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
```

Step2: train
1. with srun
```shell
g=$(($2<8?$2:8))
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
    --job-name=$cfg \
python -m up train \
  --config=$cfg \
  --display=1 \
  2>&1 | tee log.train

# ./train.sh <PARTITION> <num_gpu> <config>
./train.sh ToolChain 8 configs/det/yolox/yolox_tiny.yaml
```



2. with spring.submit
```shell
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

spring.submit run -n$1 -p spring_scheduler --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
"python -m up train \
  --config=$cfg \
  --display=10 \
  2>&1 | tee log.train "


#./train.sh <num_gpu> <PARTITION> <config> <job-name>
./train.sh 8 ToolChain configs/det/yolox/yolox_tiny.yaml yolox_tiny
```

Step3: fp16, add fp16 setting into runtime config

```yaml
runtime:
    fp16: True
```

### Eval
Step1: edit config of evaluating dataset

Step2: test

```shell
g=$(($2<8?$2:8))
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
    --job-name=$cfg \
python -m up train -e\
  --config=$3 \
  --display=1 \
  2>&1 | tee log.eval

# ./eval.sh <PARTITION> <num_gpu> <config>
./eval.sh ToolChain 1 configs/det/yolox/yolox_tiny.yaml
```


### Demo
Step1: add visualizer config in yaml

```yaml
inference:
  visualizer:
    type: plt
    kwargs:
      class_names: ['__background__', 'person'] # class names
      thresh: 0.5
``` 

Step2: inference

```shell
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
```

### Mpirun mode
UP supports **mpirun** mode to launch task, MPI needs to be installed firstly

```shell
# download mpich
wget https://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz # other versions: https://www.mpich.org/static/downloads/

tar -zxvf mpich-3.2.1.tar.gz
cd mpich-3.2.1
./configure  --prefix=/usr/local/mpich-3.2.1
make && make install
```

Launch task

```shell
mpirun -np 8 python -m up train --config configs/det/yolox/yolox_tiny.yaml --launch mpi 2>&1 | tee log.train
```

* Add mpirun -np x; x indicates number of processes
* Mpirun is convenient to debug with pdb
* --launch: mpi

## Custom Example

* [custom dataset](configs/det/custom/custom_dataset.yaml)
* [rank_dataset](configs/det/custom/rank_dataset.yaml)

## Benckmark

* [YOLOX](docs/benchmark.md) 
* [YOLOX-Ret](docs/benchmark.md)
* [EFL](docs/equalized_focal_loss.md)
* [YOLOV5](docs/benchmark.md)
* [RetinaNet](docs/benchmark.md)
* [QAT](docs/quant.md)
* [Faster-RCNN](docs/two_stages.md)

## Quick Run

* [Training on custom data](docs/train_custom_data.md)
* [Register modules](docs/register_modules.md)
* [UP developing mode](docs/up_developing_mode.md)

## Tutorials

* [Learn about configs](docs/learn_about_configs.md)
* [Datasets](docs/datasets.md)
* [Augmentations and preprocesses](docs/augmentations.md)
* [Loss functions](docs/loss_functions.md)

## Useful Tools

* [Hooks](docs/hooks.md)
* [Visualization](docs/visualization.md)


## References

* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [YOLOv5](https://github.com/ultralytics/yolov5)

## Acknowledgments

Thanks to all past contributors, especially [opcoder](https://github.com/opcoder),
