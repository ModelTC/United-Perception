## 基础模型

| model            | setting | Params(M) | FLOPS(G) | ema   | train size | bs   | epoch | test size | top-1 |
| ---------------- | ------- | --------- | -------- | ----- | ---------- | ---- | ----- | --------- | ----- |
| [resnet18](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/resnet/res18.yaml)         | Steplr  | 11.690    | 1.813    | False | 224        | 1024 | 100   | 224       | 70.13 |
| resnet34        | Steplr  | 21.798    | 3.663    | False | 224        | 1024 | 100   | 224       | 74.03 |
| [resnet50](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/resnet/res50.yaml)         | Steplr  | 25.557    | 4.087    | False | 224        | 1024 | 100   | 224       | 76.76 |
| [mobilenetv2_0.5](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/mobilenetv2/mbv2_0.5_batch1k_epoch250_coslr_nesterov_wd0.00004_bn_nowd_fp16_ema.yaml)  | coslr   | 1.969     | 0.096    | True  | 224        | 1024 | 350   | 224       | 65.36 |
| [mobilenetv2_1.0](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/mobilenetv2/mbv2_1.0_batch1k_epoch250_coslr_nesterov_wd0.00004_bn_nowd_fp16_ema.yaml)  | coslr   | 3.505     | 0.299    | True  | 224        | 1024 | 350   | 224       | 73.35 |
| [mobilenetv2_2.0](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/mobilenetv2/mbv2_2.0_batch1k_epoch250_coslr_nesterov_wd0.00004_bn_nowd_fp16_ema.yaml)  | coslr   | 11.258    | 1.134    | True  | 224        | 1024 | 350   | 224       | 77.56 |
| [mobilenetv3_s1.0](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/mobilenetv3/mbv3_small_1.0_batch1k_epoch350_coslr_nesterov_wd0.00003_bn_nowd_fp16_ema0.9999_dropout0.2.yaml) | coslr   | 2.938     | 0.054    | True  | 224        | 1024 | 350   | 224       | 67.97 |
| [mobilenetv3_l1.0](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/mobilenetv3/mbv3_large_1.0_batch1k_epoch350_coslr_nesterov_wd0.00003_bn_nowd_fp16_ema0.9999_dropout0.2.yaml) | coslr   | 5.476     | 0.209    | True  | 224        | 1024 | 350   | 224       | 75.17 |
| [regnetx200](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/regnet/reg_x200.yaml)       | coslr   | 2.685     | 0.194    | False | 224        | 1024 | 100   | 224       | 68.19 |
| [regnetx400](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/regnet/reg_x400.yaml)       | coslr   | 5.158     | 0.388    | False | 224        | 1024 | 100   | 224       | 71.98 |
| [regnety200](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/regnet/reg_y200.yaml)       | coslr   | 3.163     | 0.194    | False | 224        | 1024 | 100   | 224       | 69.96 |
| [regnety400](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/regnet/reg_y400.yaml)       | coslr   | 4.344     | 0.391    | False | 224        | 1024 | 100   | 224       | 73.41 |
| [convnext_tiny](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/convnext/convnext_t.yaml)    | coslr   | 28.6      | 4.5      | True  | 224        | 4096 | 300   | 224       | 81.22 |
| [convnext_small](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/dev/configs/cls/convnext/convnext_s.yaml)   | coslr   | 50        | 8.7      | True  | 224        | 4096 | 300   | 224       | 82.74 |
## 高精度baseLine

| model                                | setting | Params(M) | FLOPS(G) | ema   | train size | bs   | epoch | test size | top-1 |
| ------------------------------------ | ------- | --------- | -------- | ----- | ---------- | ---- | ----- | --------- | ----- |
| [resnet18+bag of tricks](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/master/configs/cls/resnet/res18_200e_bag_of_tricks.yaml)               | coslr   | 11.690    | 1.813    | False | 224        | 2048 | 200   | 224       | 70.95 |
| [resnet18+strikes](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/master/configs/cls/resnet/res18_strikes_300e_bce.yaml)                     | coslr   | 11.690    | 1.813    | True  | 224        | 2048 | 300   | 224       | 72.78 |
| resnet18+resnet152                   | step    | 11.690    | 1.813    | False | 224        | 2048 | 180   | 224       | 72.83 |
| [resnet18+resnet152+bag of tricks](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/master/configs/cls/resnet/res18_kd_bag_of_tricks.yaml)     | step    | 11.690    | 1.813    | False | 224        | 2048 | 180   | 224       | 73.03 |
| resnet50+bag of tricks               | coslr   | 25.557    | 4.087    | False | 224        | 2048 | 200   | 224       | 78.21 |
| [resnet50+strikes](https://gitlab.bj.sensetime.com/spring2/universal-perception/-/blob/master/configs/cls/resnet/res50_strikes_300e_bce.yaml)                     | coslr   | 25.557    | 4.087    | False | 224        | 2048 | 300   | 224       | 79.16 |
| resnet50-D + bag of tricks           | coslr   | 25        | 4.3      | False | 224        | 2048 | 200   | 224       | 78.9  |

## 下游baseLine

| model                                | setting | Params(M) | FLOPS(G) | ema   | train size | bs   | epoch | test size | top-1 |
| ------------------------------------ | ------- | --------- | -------- | ----- | ---------- | ---- | ----- | --------- | ----- |
| resnet50/flower                      | step    | 25.557    | 4.087    | False | 224        | 64   | 150   | 224       | 96.86 |
| resnet50/cars                        | step    | 25.557    | 4.087    | False | 224        | 64   | 150   | 224       | 92.06 |
| resnet50+strikes/flower              | step    | 25.557    | 4.087    | False | 224        | 64   | 300   | 224       | 97.26 |
| resnet50+strikes/cars                | step    | 25.557    | 4.087    | False | 224        | 64   | 300   | 224       | 93.52 |