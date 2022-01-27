# UP

![image](up-logo.png)

## 安装
```shell
source s0.3.4 or s0.3.3
./easy_setup.sh <partition>
```

### Train
Step1: 修改dataset 路径，基本格式沿袭POD的格式，可以参考POD的文档

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
1. Srun 直接启动脚本
```shell
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
```

2. Spring.submit 启动脚本
```shell
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
```

Step3: FP16 设置以及其他一些额外的设置

```yaml
runtime:
    fp16: # linklink 后端
        keep_batchnorm_fp32: True
        scale_factor: dynamic
    # fp16: True # ddp 后端
    runner:
       type: base # 默认是base，也可以根据需求注册所需的runner，比如量化quant
```

### Eval
评测脚本, 沿袭POD的模式，现在将train test 合成了一个指定，在命令行指定 -e 即可启动测试

```shell
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
```

### 部署

* tocaffe
```shell
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

```

* to_kestrel

TODO cfg example

```shell
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
```


### Demo
Step1: 修改cfg，沿袭POD的格式

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

## Custom Example

* [custom dataset](configs/det/custom/custom_dataset.yaml)
* [rank_dataset](configs/det/custom/rank_dataset.yaml)

## Benckmark

* [Det](docs/detection_benchmark.md)
* [Cls] 
* [Seg](docs/semantic_benchmark.md)

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


## 说明

分割部分主体代码来自light-seg, 分类代码来自prototype, 检测部分来自于POD
大部分文档可参考 [POD 官方文档](http://spring.sensetime.com/docs/pod/index.html)

## TODO
- [ ] Transformer 基础backbone 支持
- [ ] MultiTask 支持
- [ ] Cls 相关自监督算法支持
- [ ] Det POD 未迁移部分
- [ ] 详细的功能文档

