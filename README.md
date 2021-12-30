# EOD

![image](eod-logo.png)

Easy and Efficient Object Detector

**EOD** (**E**asy and **E**fficient **O**bject **D**etection) is a general object detection model production framework.
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

pip install -r requirments

## Get Started
Some example scripts are supported in scripts/.

### Export Module
Export eod into ROOT and PYTHONPATH

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

```shell
python -m eod train --config configs/det/yolox/yolox_tiny.yaml --nm 1 --ng 8 --launch pytorch 2>&1 | tee log.train
```
* --config: yamls in configs/
* --nm: machine number
* --ng: gpu number for each machine
* --launch: slurm or pytorch

Step3: fp16, add fp16 setting into runtime config

```yaml
runtime:
    fp16: True
```

### Eval
Step1: edit config of evaluating dataset

Step2: test

```shell
python -m eod train -e --config configs/det/yolox/yolox_tiny.yaml --nm 1 --ng 1 --launch pytorch 2>&1 | tee log.test
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
python -m eod inference --config configs/det/yolox/yolox_tiny.yaml --ckpt ckpt_tiny.pth -i imgs -v vis_dir
```
* --ckpt: model for inferencing
* -i: images directory or single image
* -v: directory saving visualization results

### Mpirun mode
EOD supports **mpirun** mode to launch task, MPI needs to be installed firstly

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
mpirun -np 8 python -m eod train --config configs/det/yolox/yolox_tiny.yaml --launch mpi 2>&1 | tee log.train
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

## Quick Run

* [Training on custom data](docs/train_custom_data.md)
* [Register modules](docs/register_modules.md)
* [EOD developing mode](docs/eod_developing_mode.md)

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