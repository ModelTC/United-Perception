## 基础模型

| model            | setting | Params(M) | FLOPS(G) | ema   | train size | bs   | epoch | test size | top-1 |
| ---------------- | ------- | --------- | -------- | ----- | ---------- | ---- | ----- | --------- | ----- |
| resnet18         | Steplr  | 11.690    | 1.813    | False | 224        | 1024 | 100   | 224       | 70.13 |
| resnet34         | Steplr  | 21.798    | 3.663    | False | 224        | 1024 | 100   | 224       | 74.03 |
| resnet50         | Steplr  | 25.557    | 4.087    | False | 224        | 1024 | 100   | 224       | 76.76 |
| mobilenetv2_0.5  | coslr   | 1.969     | 0.096    | True  | 224        | 1024 | 350   | 224       | 65.36 |
| mobilenetv2_1.0  | coslr   | 3.505     | 0.299    | True  | 224        | 1024 | 350   | 224       | 73.35 |
| mobilenetv2_2.0  | coslr   | 11.258    | 1.134    | True  | 224        | 1024 | 350   | 224       | 77.56 |
| mobilenetv3_s1.0 | coslr   | 2.938     | 55.18    | True  | 224        | 1024 | 350   | 224       | 67.97 |
| mobilenetv3_l1.0 | coslr   | 5.476     | 213.80   | True  | 224        | 1024 | 350   | 224       | 75.17 |
| regnetx200       | coslr   | 2.685     | 198.67   | False | 224        | 1024 | 100   | 224       | 68.19 |
| regnetx400       | coslr   | 5.158     | 397.29   | False | 224        | 1024 | 100   | 224       | 71.98 |
| regnety200       | coslr   | 3.163     | 199.15   | False | 224        | 1024 | 100   | 224       | 69.96 |
| regnety400       | coslr   | 4.344     | 401.40   | False | 224        | 1024 | 100   | 224       | 73.41 |

## 高精度baseLine

| model                                | setting | Params(M) | FLOPS(G) | ema   | train size | bs   | epoch | test size | top-1 |
| ------------------------------------ | ------- | --------- | -------- | ----- | ---------- | ---- | ----- | --------- | ----- |
| resnet18+bag of tricks               | coslr   | 11.690    | 1.813    | False | 224        | 2048 | 200   | 224       | 70.95 |
| resnet18+strikes                     | coslr   | 11.690    | 1.813    | True  | 224        | 2048 | 300   | 224       | 72.78 |
| resnet18+resnet152                   | step    | 11.690    | 1.813    | False | 224        | 2048 | 180   | 224       | 72.83 |
| resnet50+bag of tricks               | coslr   | 25.557    | 4.087    | False | 224        | 2048 | 200   | 224       | 78.35 |
| resnet50-D + bag of tricks(no mixup) | coslr   | 25        | 4.3      | False | 224        | 2048 | 200   | 224       | 77.5  |
| resnet50-D + bag of tricks           | coslr   | 25        | 4.3      | False | 224        | 2048 | 200   | 224       | 78.9  |

