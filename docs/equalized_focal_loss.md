# Equalized Focal Loss

This repo is about the Equalized Focal Loss (EFL)  and the Equalized Quality Focal Loss (EQFL) for one-stage long-tailed object detection.

## Requirements

- Python 3.6+
- Pytorch 1.5.0+
- CUDA 9.0+
- [EOD](https://github.com/ModelTC/EOD)

## Prepare Dataset (LVIS v1)
### images and annotations

LVIS v1 uses same images as COCO's. Thus you only need to download COCO dataset at folder ($COCO) and link its `train`, `val`, and `test` to folder ($LVIS).
```
# download $COCO/train, $COCO/val, and $COCO/test, firstly
mkdir $LVIS
ln -s $COCO/train $LVIS
ln -s $COCO/val $LVIS
ln -s $COCO/test $LVIS
```
Then download the annotations from the official website of [LVIS](https://www.lvisdataset.org/dataset). The annotations should be placed at ($LVIS/annotations).
```
cd $LVIS
mkdir annotations
# then download the annotations of lvis_v1_train.json and lvis_v1_val.json
```
Finally the file structure of folder ($LVIS) will be like this:
```
$LVIS
  ├── annotations
  │   ├── lvis_v1_val.json
  │   ├── lvis_v1_train.json
  ├── train2017
  │   ├── 000000004134.png
  │   ├── 000000031817.png
  │   ├── ......
  ├── val2017
  ├── test2017
```
### configs

Modify the dataset config like this:
```
dataset:
  train:
    dataset:
      type: lvisv1
      kwargs:
        meta_file: $LVIS/annotations/lvis_v1_train.json
        image_reader:
          type: fs_opencv
          kwargs:
            image_dir: $LVIS
            color_mode: BGR
        ...
  test:
    dataset:
      type: lvisv1
      kwargs:
        meta_file: &gt_file $LVIS/annotations/lvis_v1_val.json
        image_reader:
          type: fs_opencv
          kwargs:
            image_dir: $LVIS
            color_mode: BGR
        ...
        evaluator:
          type: LVIS
          kwargs:
            gt_file: *gt_file
            iou_types: [bbox]
```

## Benchmark Results

We provide the benchmark results of the EFL (Equalized Focal Loss) and the EQFL (Equalized Quality Focal Loss).
The results are divided into the improved baseline series and the YOLOX* series (YOLOX trained with our improved settings).
**All models are trained with the repeat factor sampler (RFS) with 16 GPUs settings**.

**Improved Baseline Series**

|config  | loss | pretrain | scheduler | AP | APr | APc | APf | weights |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
|[Res50](https://github.com/ModelTC/EOD/blob/main/configs/det/efl/efl_improved_baseline_r50_2x_rfs.yaml)| EFL | imagenet | 24e | 27.5 | 20.2 | 26.1 | 32.4 | [model](https://github.com/ModelTC/EOD/releases/download/0.1.0/efl_improved_baseline_r50.pth) |
|[Res101](https://github.com/ModelTC/EOD/blob/main/configs/det/efl/efl_improved_baseline_r50_2x_rfs.yaml) | EFL | imagenet | 24e | 29.2 | 23.5 | 27.4 | 33.8 | [model](https://github.com/ModelTC/EOD/releases/download/0.1.0/efl_improved_baseline_r101.pth) |

**YOLOX\* Series**

|config  | loss | pretrain | scheduler | AP | APr | APc | APf | weights |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
|[YOLOX-S*](https://github.com/ModelTC/EOD/blob/main/configs/det/efl/efl_yolox_small.yaml)| EFL | None | 300e | 23.3 | 18.1 | 21.2 | 28.0 | [model](https://github.com/ModelTC/EOD/releases/download/0.1.0/efl_yolox_small.pth) |
|[YOLOX-S*](https://github.com/ModelTC/EOD/blob/main/configs/det/efl/eqfl_yolox_small.yaml)| EQFL | None | 300e | 24.2 | 16.3 | 22.7 | 29.4 | [model](https://github.com/ModelTC/EOD/releases/download/0.1.0/eqfl_yolox_small.pth) |
|[YOLOX-M*](https://github.com/ModelTC/EOD/blob/main/configs/det/efl/efl_yolox_medium.yaml)| EFL | None | 300e | 30.0 | 23.8 | 28.2 | 34.7 | [model](https://github.com/ModelTC/EOD/releases/download/0.1.0/efl_yolox_medium.pth) |
|[YOLOX-M*](https://github.com/ModelTC/EOD/blob/main/configs/det/efl/eqfl_yolox_medium.yaml)| EQFL | None | 300e | 31.0 | 24.0 | 29.1 | 36.2 | [model](https://github.com/ModelTC/EOD/releases/download/0.1.0/eqfl_yolox_medium.pth) |

## Testing with Pretrained Models

For example, if you want to test the pretrained model of YOLOX-M* with the EQFL (Equlized Quailty Focal Loss):
```
mkdir pretrain
# download the weight of YOLOX-M* with the EQFL (eqfl_yolox_medium.pth)
``` 
Edit the saver config of `configs/det/efl/eqfl_yolox_medium.yaml`:
```
saver:
  save_dir: checkpoints/yolox_medium
  pretrain_model: pretrain/eqfl_yolox_medium.pth
  results_dir: results_dir/yolox_medium
  auto_resume: True
```
Then run the following commond to evaluate.
```
python -m eod train -e --config configs/det/efl/eqfl_yolox_medium.yaml --nm 1 --ng 1 --launch pytorch 2>&1 | tee log.test
```

## Training

It should be notice that our benchmark results are all come from 16 GPUs settings on the Nvidia A100. Since the 16 GPUs settings is not always available for all readers, we provide the 8 GPUs settings here. For example, you can run the following commond to train a model of the improved baseline with EFL (Res50):
```
# please change the batch_size to 2 of the train batch_sampler in the 
# configs/det/efl/efl_improved_baseline_r50_2x_rfs.yaml, firstly.

python -m eod train --config configs/det/efl/efl_improved_baseline_r50_2x_rfs.yaml --nm 1 --ng 8 --launch pytorch 2>&1 | tee log.train
```
Meanwhile, the gradient_collector hook in the configs of EFL and EQFL is of vital importance on the gradient collection mechanism. Please make sure it is open when train networks with these loss.
```
hooks:
  - type: gradient_collector
    kwargs:
      hook_head: roi_head          # the classification head
      hook_cls_head: cls_preds     # the last layer of the classification head
```
- Tip 1: our gradient collection mechanism only support the `fp32` settings. Training with `fp16` will get unexpected results.
- Tip 2: an alternate approach to obtain the gradient is by manual calculation just like the way in [EQLv2](https://github.com/tztztztztz/eqlv2/blob/master/mmdet/models/losses/eqlv2.py#L90). One thing need to be noticed is that you need to write you own code to calculate the gradient from the derivative of EFL or EQFL.


## How to Train EFL on OpenImages

Our EOD framework support the training and inference of OIDs (but can not evaluate the results directly). Most of our steps are following the guide in [EQLv2](https://github.com/tztztztztz/eqlv2#how-to-train-eqlv2-on-openimages). 

Here are the steps to get a result of OIDs (the bold steps are the operations that need to be noticed, other steps are same as them in EQLv2):
- Download the data.
- Convert the `.csv` to coco-like `.json` file.
- **Train and inference the models**
    ```
    # edit image and annotation paths in 
    # configs/det/efl/efl_oids_r50_2x_random.yaml, firstly.
    # then change the batch_size to 2 for 8 GPUs settings.
    # we train the model with a 120k/160k/180k scheduler with random sampler.

    python -m eod train --config configs/det/efl/efl_oids_r50_2x_random.yaml --nm 1 --ng 8 --launch pytorch 2>&1 | tee log.train
    ```
- **Obtain the inference results `results.txt.all` under the `results_dir`**
- Convert coco-like `results.txt.all` result file to openimage like `.csv` file
- Evaluate results file using official API
- Parse the AP file and output the grouped AP

Our OIDs results are:
```
# Res50
mAP 0.5151686921063541
mAP0: 0.5277291485206159
mAP1: 0.5289892622419364
mAP2: 0.5082855060435439
mAP3: 0.5017385315742882
mAP4: 0.5094778258438134

# Res101
mAP 0.5255129979062584
mAP0: 0.53395152741142
mAP1: 0.5381753139992043
mAP2: 0.5139450335860504
mAP3: 0.5182665336563826
mAP4: 0.5234797367633899
```
