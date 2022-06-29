# keypoint benchmark
| Network                                  | Dataset | Input Size |bs      | Loss & Label            | bbox(AP 51) | model |
| ---------------------------------------- | ------- | ---------- | ------ |----------------------- | ----------- | ----- |      
| ResNet-50 Simple Baseline(hkd flip=False) | COCO    | 256*192   | 64 * 8 |MSE & Gaussian(sigma=2) | 69.4        | -    |
| [ResNet-50 Simple Baseline](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/kp/res50_sb.yaml)                | COCO    | 256*192    | 64 * 8  | MSE & Gaussian(sigma=2) | 69.5        | [ckpt](http://spring.sensetime.com/dropadmin/$/x3SOS.pth)      |
| ResNet-50 FPN   (hkd flip=False)   | COCO    | 256*192    |      64 * 8  | SCE & radius 0, 0, 1, 2 | 68.8        |   -   |
| [ResNet-50 FPN](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/kp/res50_fpn.yaml)                     | COCO    | 256*192   | 64 * 8 | SCE & radius 0, 0, 1, 2 | 70.7        | [ckpt](http://spring.sensetime.com/dropadmin/$/CZANT.pth)     |
| hourglass 2-stack   (hkd flip=False)      | COCO    | 256*256    | 12 * 8| SCE & radius 1, 1       | 70.3        |    -  |
| [hourglass 2-stack](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/kp/hg2.yaml)                        | COCO    | 256*256    | 12 * 8| SCE & radius 1, 1       | 72.4        | [ckpt](http://spring.sensetime.com/dropadmin/$/3wBcd.pth)   |
| [xmnet](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/kp/xmnet_sb.yaml)                        | COCO    | 256*192    | 64 * 16| MSE & Gaussian(sigma=2)       | 66.3        | [ckpt](http://spring.sensetime.com/dropadmin/$/lnzHX.pth)   |
