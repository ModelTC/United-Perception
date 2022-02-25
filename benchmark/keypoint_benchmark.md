# keypoint benchmark
| Network                                  | Dataset | Input Size | Loss & Label            | bbox(AP 51) | model |
| ---------------------------------------- | ------- | ---------- | ----------------------- | ----------- | ----- |      
| ResNet-50 Simple Baseline(hkd flip=False) | COCO    | 256*192    | MSE & Gaussian(sigma=2) | 69.4        | -    |
| [ResNet-50 Simple Baseline](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/kp/res50_sb.yaml)                | COCO    | 256*192    | MSE & Gaussian(sigma=2) | 69.5        | [ckpt](http://spring.sensetime.com/dropadmin/$/x3SOS.pth)      |
| ResNet-50 FPN   (hkd flip=False)   | COCO    | 256*192    | SCE & radius 0, 0, 1, 2 | 68.8        |   -   |
| [ResNet-50 FPN](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/kp/res50_fpn.yaml)                     | COCO    | 256*192    | SCE & radius 0, 0, 1, 2 | 70.7        | [ckpt](http://spring.sensetime.com/dropadmin/$/CZANT.pth)     |
| hourglass 2-stack   (hkd flip=False)      | COCO    | 256*256    | SCE & radius 1, 1       | 70.3        |    -  |
| [hourglass 2-stack](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/kp/hg2.yaml)                        | COCO    | 256*256    | SCE & radius 1, 1       | 72.2        | [ckpt]()   |